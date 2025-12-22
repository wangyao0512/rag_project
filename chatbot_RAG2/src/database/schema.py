"""
SQLite database schema for medical knowledge
"""
import sqlite3
from typing import List, Dict, Optional
import json
from datetime import datetime


class MedicalDatabase:
    def __init__(self, db_path="database/medical.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        """Create lightweight tables"""
        cursor = self.conn.cursor()

        # Main evidence table (simplified)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_type TEXT,
                disease TEXT,
                intervention TEXT,
                drug_a TEXT,
                drug_b TEXT,
                population TEXT,
                outcome TEXT,
                evidence_level TEXT,
                chunk_ids TEXT,  -- JSON array of chunk IDs
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Text chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,  -- JSON metadata
                source_doc TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Entity normalization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_mappings (
                raw_term TEXT PRIMARY KEY,
                normalized_term TEXT,
                entity_type TEXT,
                synonyms TEXT  -- JSON array
            )
        ''')

        # Create indexes for fast search
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_disease ON medical_evidence(disease)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intervention ON medical_evidence(intervention)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_drugs ON medical_evidence(drug_a, drug_b)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evidence_type ON medical_evidence(evidence_type)')

        self.conn.commit()

    def insert_evidence(self, evidence: Dict) -> int:
        """Insert medical evidence"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO medical_evidence
            (evidence_type, disease, intervention, drug_a, drug_b,
             population, outcome, evidence_level, chunk_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evidence.get('evidence_type'),
            evidence.get('disease'),
            evidence.get('intervention'),
            evidence.get('drug_a'),
            evidence.get('drug_b'),
            evidence.get('population'),
            evidence.get('outcome'),
            evidence.get('evidence_level'),
            json.dumps(evidence.get('chunk_ids', []))
        ))
        self.conn.commit()
        return cursor.lastrowid

    def insert_chunk(self, chunk_id: str, content: str, metadata: Dict, source_doc: str):
        """Insert text chunk"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO text_chunks (chunk_id, content, metadata, source_doc) VALUES (?, ?, ?, ?)",
            (chunk_id, content, json.dumps(metadata), source_doc)
        )
        self.conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM text_chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def search_evidence(self, disease: str = None, intervention: str = None,
                       drug: str = None, limit: int = 10) -> List[Dict]:
        """Search medical evidence"""
        cursor = self.conn.cursor()

        conditions = []
        params = []

        if disease:
            conditions.append("disease LIKE ?")
            params.append(f"%{disease}%")

        if intervention:
            conditions.append("intervention LIKE ?")
            params.append(f"%{intervention}%")

        if drug:
            conditions.append("(drug_a LIKE ? OR drug_b LIKE ?)")
            params.extend([f"%{drug}%", f"%{drug}%"])

        if not conditions:
            query = "SELECT * FROM medical_evidence LIMIT ?"
            params = [limit]
        else:
            query = f"SELECT * FROM medical_evidence WHERE {' AND '.join(conditions)} LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def insert_entity_mapping(self, raw_term: str, normalized_term: str,
                            entity_type: str, synonyms: List[str] = None):
        """Insert entity mapping"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO entity_mappings (raw_term, normalized_term, entity_type, synonyms) VALUES (?, ?, ?, ?)",
            (raw_term, normalized_term, entity_type, json.dumps(synonyms or []))
        )
        self.conn.commit()

    def get_normalized_term(self, raw_term: str) -> Optional[str]:
        """Get normalized term"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT normalized_term FROM entity_mappings WHERE raw_term = ?", (raw_term,))
        row = cursor.fetchone()
        if row:
            return row['normalized_term']
        return None

    def close(self):
        """Close database connection"""
        self.conn.close()

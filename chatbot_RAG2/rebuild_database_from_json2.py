"""
Rebuild medical database from processed JSON files in 1215_MVP_processed_final
Converts 'score' field to 'grade' field
"""
import re
import sys
import os
import json
import uuid
from pathlib import Path
from loguru import logger
from config import Config
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.database.schema import MedicalDatabase
from src.retrieval.vector_store import VectorStore

logger.remove()
logger.add(sys.stderr, level="INFO")
def remove_parentheses_content(text):
    text = re.sub(r'\([^)]*\)$', '', text)
    text = re.sub(r'（[^)]*）$', '', text)
    text = re.sub(r'\<[^)]*\>$', '', text)
    return text.strip()
def rebuild_database():
    """Rebuild complete database from JSON files"""

    json_folder = Path("/data_vision9/ywang/heisesuliuMVP_final_1209_processed_dedeup0_99/unique")

    logger.info("="*80)
    logger.info("REBUILDING MEDICAL DATABASE FROM JSON FILES")
    logger.info("="*80)

    # Check JSON folder exists
    if not json_folder.exists():
        logger.error(f"JSON folder not found: {json_folder}")
        return False

    # Count JSON files
    json_files = list(json_folder.glob("*_pico.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")

    if len(json_files) == 0:
        logger.error("No JSON files found!")
        return False

    # Initialize new databases
    logger.info("\nInitializing new databases...")
    os.makedirs(Config.DATABASE_DIR, exist_ok=True)
    db = MedicalDatabase(Config.DATABASE_DIR / "medical.db")
    vector_store = VectorStore(persist_dir=Config.DATABASE_DIR / "chroma")
    logger.info("✓ Databases initialized")

    # Process all JSON files
    total_entries = 0
    total_chunks = 0
    all_texts = []
    all_metadatas = []
    all_ids = []
    chunk_to_evidence = {}  # Map chunk_id to evidence_id

    logger.info("\nProcessing JSON files...")
    for idx, json_file in enumerate(json_files, 1):
        # Skip macOS resource fork files
        if json_file.name.startswith('._'):
            continue

        logger.info(f"[{idx}/{len(json_files)}] Processing: {json_file.name}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry in entries:
                # Convert 'score' to 'grade'
                grade = entry.get('score', entry.get('grade', 10))

                # Create chunk for this entry first
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                # 优先选择llm_text，而不是original_text
                chunk_text = entry.get('llm_text', entry.get('original_text', ''))
                # p_cleaned = remove_parentheses_content(entry.get('P', ''))
                # i_cleaned = remove_parentheses_content(entry.get('I', ''))
                # c_cleaned = remove_parentheses_content(entry.get('C', ''))
                # o_cleaned = remove_parentheses_content(entry.get('O', ''))

                # chunk_text = ', '.join([text for text in [p_cleaned, i_cleaned, c_cleaned, o_cleaned] if text])
                
                # chunk_text = ','.join([chunk_text,entry.get('llm_text', entry.get('original_text', ''))])
                # Prepare evidence data with grade field
                evidence_data = {
                    "evidence_type": "pico",
                    "disease": entry.get('P', ''),
                    "intervention": entry.get('I', ''),
                    "drug_a": None,
                    "drug_b": None,
                    "comparison": entry.get('C', ''),
                    "outcome": entry.get('O', ''),
                    "evidence_level": entry.get('confidence', 'high'),
                    "chunk_ids": [chunk_id],
                    "grade": grade,  # Use grade instead of score
                    "guideline_focus": entry.get('guideline_focus', 'Treatment'),
                    "source": entry.get('reference', 'Unknown'),
                    "year": entry.get('year', 2025)
                }

                # Insert evidence into database
                evidence_id = db.insert_evidence(evidence_data)
                total_entries += 1

                # Prepare chunk metadata
                chunk_metadata = {
                    "original_text":entry.get('original_text', 'Unknown'),
                    "original_text_all":entry.get('original_text_all', 'Unknown'),
                    "source": entry.get('reference', 'Unknown'),
                    "year": entry.get('year', 2025),
                    "grade": grade,
                    "guideline_focus": entry.get('guideline_focus', 'Treatment'),
                    "publish_organization": entry.get('publish_organization', 'Unknown'),
                    "confidence": entry.get('confidence', 'high')
                }

                # Insert chunk into database
                db.insert_chunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=chunk_metadata,
                    source_doc=entry.get('reference', 'Unknown')
                )

                # Collect for vector embedding
                all_texts.append(chunk_text)
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)
                chunk_to_evidence[chunk_id] = evidence_id
                total_chunks += 1
            
            output_path = Path("all_texts_1209_processed_llm.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_texts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            continue

    logger.info(f"\n✓ Processed {total_entries} evidence entries")
    logger.info(f"✓ Created {total_chunks} text chunks")

    # Add to vector store in batches
    logger.info("\nIndexing chunks in vector store...")
    batch_size = 4
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]

        vector_store.add_documents(batch_texts, batch_metadatas, batch_ids)
        logger.info(f"  Indexed batch {i//batch_size + 1}/{(len(all_texts)-1)//batch_size + 1}")

    logger.info(f"✓ Vector store indexed with {vector_store.count()} embeddings")

    # Verification
    logger.info("\n" + "="*80)
    logger.info("DATABASE REBUILD COMPLETE!")
    logger.info("="*80)
    logger.info(f"  - Evidence entries: {total_entries}")
    logger.info(f"  - Text chunks: {total_chunks}")
    logger.info(f"  - Vector embeddings: {vector_store.count()}")
    logger.info(f"  - Database file: {Config.DATABASE_DIR / 'medical.db'}")
    logger.info(f"  - Vector store: {Config.DATABASE_DIR / 'chroma'}")
    logger.info("="*80 + "\n")

    return True


if __name__ == "__main__":
    success = rebuild_database()
    sys.exit(0 if success else 1)

"""
Ingest PICO data from 12.15_MVP folder into the medical RAG system
"""
import sys
import json
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import MedicalRAG

def ingest_pico_file(rag, pico_file_path):
    """Ingest a single PICO JSON file"""
    logger.info(f"Processing {pico_file_path.name}...")

    with open(pico_file_path, 'r', encoding='utf-8') as f:
        pico_data = json.load(f)

    if not isinstance(pico_data, list):
        logger.error(f"Invalid format in {pico_file_path.name}, expected list")
        return 0

    count = 0
    for idx, pico in enumerate(pico_data, 1):
        try:
            # Extract reference/source
            reference = pico.get('reference', '未知来源')

            # Extract year from reference
            import re
            year_match = re.search(r'(20\d{2})', reference)
            year = year_match.group(1) if year_match else ''

            # Create evidence entry
            evidence = {
                'evidence_type': 'pico',
                'disease': '多种疾病',  # General, will be extracted from P/I/O
                'intervention': pico.get('I', ''),
                'population': pico.get('P', ''),
                'comparison': pico.get('C', ''),
                'outcome': pico.get('O', ''),
                'source': reference,
                'year': pico.get('year', year),  # Use year field if available
                'confidence': pico.get('confidence', ''),
                'note': pico.get('note', ''),
                'grade': pico.get('score'),  # Numeric GRADE (1-10)
                'cluster': pico.get('cluster', '')  # Evidence cluster
            }

            # Create text chunks from both original and LLM-generated text
            chunks = []

            # Original text chunk
            if pico.get('original_text'):
                chunks.append({
                    'content': pico['original_text'],
                    'metadata': {
                        'source': reference,
                        'year': pico.get('year', year),
                        'type': 'original',
                        'pico_element': 'full',
                        'grade': pico.get('score'),  # Add GRADE to metadata
                        'cluster': pico.get('cluster', '')  # Add cluster to metadata
                    }
                })

            # LLM-generated summary chunk
            if pico.get('llm_text'):
                chunks.append({
                    'content': pico['llm_text'],
                    'metadata': {
                        'source': reference,
                        'year': pico.get('year', year),
                        'type': 'llm_summary',
                        'pico_element': 'full',
                        'P': pico.get('P', ''),
                        'I': pico.get('I', ''),
                        'C': pico.get('C', ''),
                        'O': pico.get('O', ''),
                        'grade': pico.get('score'),  # Add GRADE to metadata
                        'cluster': pico.get('cluster', '')  # Add cluster to metadata
                    }
                })

            # Add to RAG system
            if chunks:
                rag.add_knowledge(evidence, chunks)
                count += 1

        except Exception as e:
            logger.error(f"Error processing PICO entry {idx} in {pico_file_path.name}: {e}")
            continue

    logger.info(f"Added {count} PICO entries from {pico_file_path.name}")
    return count


def main():
    logger.info("Initializing Medical RAG system...")

    # Initialize RAG system
    rag = MedicalRAG()

    # Get all PICO JSON files from 12.15_MVP_2 folder
    mvp_folder = Path("/data1_vision14/ywang/Meta/chatbot_RAG/code/medical-rag-prototype/data/processed_papers/1215_MVP_processed_final")
    pico_files = list(mvp_folder.glob("*.json"))

    # Filter out _pico_origin.json files and macOS hidden files
    pico_files = [f for f in pico_files
                  if not f.name.endswith('_pico_origin.json')
                  and not f.name.startswith('._')]

    logger.info(f"Found {len(pico_files)} PICO files to process")

    total_count = 0
    total_chunks = 0

    for pico_file in pico_files:
        count = ingest_pico_file(rag, pico_file)
        total_count += count

    # Get final counts from database
    db_count = rag.db.conn.execute("SELECT COUNT(*) FROM medical_evidence").fetchone()[0]
    chunk_count = rag.db.conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
    vector_count = rag.vector_store.count()

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"Total PICO entries processed: {total_count}")
    logger.info(f"Total evidence entries in DB: {db_count}")
    logger.info(f"Total text chunks in DB: {chunk_count}")
    logger.info(f"Total vector embeddings: {vector_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

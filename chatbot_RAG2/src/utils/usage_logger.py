"""
Thread-safe usage logging for medical RAG system
Records user queries, responses, and citations to a log file
"""
import json
import fcntl
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class UsageLogger:
    """Thread-safe logger for recording user interactions"""

    def __init__(self, log_file: str = "logs/usage_log.txt"):
        """
        Initialize usage logger

        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Usage log directory ensured: {log_dir}")

    def log_interaction(
        self,
        question: str,
        answer: str,
        raw_answer: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        entities: Optional[Dict] = None,
        cot_analysis: Optional[Dict] = None,
        session_id: Optional[str] = None
    ):
        """
        Log a user interaction to the usage log file (thread-safe)

        Args:
            question: User's question
            answer: Clean answer text (without thinking tags)
            raw_answer: Raw answer including thinking tags (optional)
            sources: List of source dictionaries with content and metadata
            entities: Extracted entities dictionary
            cot_analysis: Chain-of-thought analysis results
            session_id: Optional session identifier for tracking conversations
        """
        # Prepare log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'question': question,
            'answer': answer,
        }

        # Add raw answer with thinking if available
        if raw_answer and raw_answer != answer:
            log_entry['answer_with_thinking'] = raw_answer

        # Add entities if available
        if entities:
            log_entry['entities'] = entities

        # Add CoT analysis if available
        if cot_analysis:
            log_entry['cot_analysis'] = cot_analysis

        # Process and add sources with citations
        if sources:
            cited_sources = []
            for idx, source in enumerate(sources, 1):
                metadata = source.get('metadata', {})
                cited_sources.append({
                    'citation_number': idx,
                    'title': metadata.get('source', '未知来源'),
                    'year': metadata.get('year', ''),
                    'paragraph': source.get('content', '')
                })
            log_entry['cited_sources'] = cited_sources
            log_entry['num_sources'] = len(cited_sources)

        # Write to file with file locking for thread safety
        try:
            # Convert to JSON string
            log_line = json.dumps(log_entry, ensure_ascii=False) + '\n'

            # Open file in append mode and acquire exclusive lock
            with open(self.log_file, 'a', encoding='utf-8') as f:
                # Acquire exclusive lock (blocks until lock is available)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Write log entry
                    f.write(log_line)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure it's written to disk
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug(f"Logged interaction to {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def get_stats(self) -> Dict:
        """
        Get statistics about logged interactions

        Returns:
            Dictionary with statistics
        """
        try:
            if not os.path.exists(self.log_file):
                return {
                    'total_interactions': 0,
                    'file_size_mb': 0
                }

            # Count lines in log file
            with open(self.log_file, 'r', encoding='utf-8') as f:
                total_interactions = sum(1 for _ in f)

            # Get file size
            file_size = os.path.getsize(self.log_file) / (1024 * 1024)  # MB

            return {
                'total_interactions': total_interactions,
                'file_size_mb': round(file_size, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {
                'total_interactions': 0,
                'file_size_mb': 0,
                'error': str(e)
            }


# Global singleton instance
_usage_logger = None


def get_usage_logger(log_file: str = "logs/usage_log.txt") -> UsageLogger:
    """
    Get or create the global usage logger instance

    Args:
        log_file: Path to the log file (only used on first call)

    Returns:
        UsageLogger instance
    """
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger(log_file)
    return _usage_logger

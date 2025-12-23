"""
Global configuration for Medical RAG (OpenAI/Qwen HTTP backend)
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    DATABASE_DIR = BASE_DIR / "database_PICO2nd_1209_dedeup0_99"
    DATABASE_DIR_LLM = BASE_DIR / "database_PICO2nd_1209_dedeup0_99"
    DATABASE_DIR_PICO = BASE_DIR / "database_PICO2nd_1209_dedeup0_99_llm"

    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR_results = BASE_DIR / "logs" / "results_PICO2nd_1209_dedeup0_99.text"
    DATA_DIR = BASE_DIR / "data"
    # ====== 模型 & OpenAI 兼容接口配置 ======

    # Chat 模型（Qwen 大模型）
    CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "http://10.6.52.31:1025/v1")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
    CHAT_API_KEY = os.getenv("CHAT_API_KEY", os.getenv("OPENAI_API_KEY", "-"))

    # Embedding 模型（Qwen3-Embedding-8B）
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://10.6.12.215:6091/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")

    # API Key（自托管一般无所谓，保持 dummy 即可）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "-")
    ENTITY_BASE_URL = os.getenv("ENTITY_BASE_URL", CHAT_BASE_URL)
    ENTITY_MODEL = os.getenv("ENTITY_MODEL", CHAT_MODEL)
    MAX_CONTEXT_LENGTH = int(os.getenv("MEDICAL_RAG_MAX_CONTEXT", "4096"))
    MAX_NEW_TOKENS = int(os.getenv("MEDICAL_RAG_MAX_TOKENS", "1024"))
    # Tooling
    ENABLE_MELANOMA_STAGING = os.getenv("MEDICAL_RAG_ENABLE_MELANOMA_STAGING", "True").lower() in ("true", "1", "yes")
    # Feature flags
    ENABLE_COT = os.getenv("MEDICAL_RAG_ENABLE_COT", "False").lower() in ("true", "1", "yes")
    ENABLE_ENTITY_EXTRACTION = os.getenv("MEDICAL_RAG_ENABLE_ENTITIES", "False").lower() in ("true", "1", "yes")
    ENABLE_TWO_STAGE = os.getenv("MEDICAL_RAG_ENABLE_TWO_STAGE", "False").lower() in ("true", "1", "yes")
    TWO_STAGE_MAX_PRIMARY = int(os.getenv("MEDICAL_RAG_TWO_STAGE_MAX_PRIMARY", "5"))
    TWO_STAGE_MAX_SUPPORTING = int(os.getenv("MEDICAL_RAG_TWO_STAGE_MAX_SUPPORTING", "3"))

    # Memory settings
    BATCH_SIZE = int(os.getenv("MEDICAL_RAG_BATCH_SIZE", "1"))
    CHUNK_SIZE = int(os.getenv("MEDICAL_RAG_CHUNK_SIZE", "500"))
    MAX_CHUNKS_PER_QUERY = int(os.getenv("MEDICAL_RAG_MAX_CHUNKS", "5"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("MEDICAL_RAG_EMBED_BATCH", "32"))

    # Database settings
    # DB_PATH = os.getenv(
    #     "MEDICAL_RAG_DB_PATH",
    #     str(DATABASE_DIR / "medical.db")
    # )
    # VECTOR_DB_PATH = os.getenv(
    #     "MEDICAL_RAG_VECTOR_DB_PATH",
    #     str(DATABASE_DIR / "chroma")
    # )
    LLM_DB_PATH = os.getenv(
        "MEDICAL_RAG_DB_PATH",
        str(DATABASE_DIR_LLM / "medical.db")
    )
    LLM_VECTOR_DB_PATH = os.getenv(
        "MEDICAL_RAG_VECTOR_DB_PATH",
        str(DATABASE_DIR_LLM / "chroma")
    )

    PICO_DB_PATH = os.getenv(
        "MEDICAL_RAG_DB_PATH",
        str(DATABASE_DIR_PICO / "medical.db")
    )
    PICO_VECTOR_DB_PATH = os.getenv(
        "MEDICAL_RAG_VECTOR_DB_PATH",
        str(DATABASE_DIR_PICO / "chroma")
    )

    # Performance
    NUM_THREADS = int(os.getenv("MEDICAL_RAG_THREADS", str(min(8, os.cpu_count() or 4))))
    USE_GPU = os.getenv("MEDICAL_RAG_USE_GPU", "False").lower() in ("true", "1", "yes")
    # 最大上下文 / 输出长度（给 prompt 用的逻辑值，不再限制本地 gguf）
    MAX_CONTEXT_LENGTH = 8192
    MAX_NEW_TOKENS = 4096

    # ====== RAG 内部参数（保持原来风格） ======
    # 向量检索
    # Gradio settings
    QUEUE_SIZE = int(os.getenv("MEDICAL_RAG_QUEUE_SIZE", "1"))
    MAX_BATCH_SIZE = int(os.getenv("MEDICAL_RAG_MAX_BATCH", "1"))
    SERVER_PORT = int(os.getenv("MEDICAL_RAG_PORT", "7862"))
    SERVER_NAME = os.getenv("MEDICAL_RAG_HOST", "0.0.0.0")
    SHARE_GRADIO = os.getenv("MEDICAL_RAG_SHARE", "False").lower() in ("true", "1", "yes")

    # Logging
    LOG_LEVEL = os.getenv("MEDICAL_RAG_LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("MEDICAL_RAG_LOG_FILE", str(LOGS_DIR / "app.log"))
    LOG_ROTATION = os.getenv("MEDICAL_RAG_LOG_ROTATION", "10 MB")

    # RAG settings
    VECTOR_SEARCH_TOP_K = int(os.getenv("MEDICAL_RAG_TOP_K", "10"))
    STRUCTURED_SEARCH_LIMIT = int(os.getenv("MEDICAL_RAG_STRUCT_LIMIT", "10"))
    MAX_CONTEXT_SOURCES = int(os.getenv("MEDICAL_RAG_MAX_SOURCES", "15"))
    DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.003"))

    # Feature flags
    ENABLE_COT = os.getenv("MEDICAL_RAG_ENABLE_COT", "False").lower() in ("true", "1", "yes")
    ENABLE_ENTITY_EXTRACTION = os.getenv("MEDICAL_RAG_ENABLE_ENTITIES", "False").lower() in ("true", "1", "yes")

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [cls.MODELS_DIR, cls.DATABASE_DIR, cls.LOGS_DIR, cls.DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration as dictionary"""
        return {
            "main_model_path": cls.MODEL_PATH,
            "embedding_model_path": cls.EMBEDDING_MODEL_PATH,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
            "max_new_tokens": cls.MAX_NEW_TOKENS,
            "num_threads": cls.NUM_THREADS,
            "use_gpu": cls.USE_GPU
        }

    @classmethod
    def get_rag_config(cls) -> dict:
        """Get RAG configuration as dictionary"""
        return {
            "vector_search_top_k": cls.VECTOR_SEARCH_TOP_K,
            "structured_search_limit": cls.STRUCTURED_SEARCH_LIMIT,
            "max_context_sources": cls.MAX_CONTEXT_SOURCES,
            "max_chunks_per_query": cls.MAX_CHUNKS_PER_QUERY,
            "enable_cot": cls.ENABLE_COT,
            "enable_entity_extraction": cls.ENABLE_ENTITY_EXTRACTION
        }


# Ensure directories exist on import
Config.ensure_directories()

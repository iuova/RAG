"""Project-wide configuration values for the local RAG pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# Base paths ---------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"
MODELS_DIR: Path = BASE_DIR / "models"
LOG_DIR: Path = BASE_DIR / "logs"
EXAMPLES_DIR: Path = BASE_DIR / "examples"

# Default resources --------------------------------------------------------
DEFAULT_JSONL: Path = DATA_DIR / "data.jsonl"
DEFAULT_COLLECTION_NAME: str = "docs"
DEFAULT_EMBEDDING_MODEL: str = os.environ.get(
    "RAG_EMBEDDING_MODEL", str(MODELS_DIR / "BAAI-bge-small-en-v1.5" / "snapshots" / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
)
DEFAULT_MODEL_FILENAME: str = os.environ.get(
    "RAG_LLM_FILENAME", "qwen2.5-7b-instruct-q4_k_m.gguf"
)

# Runtime tuning -----------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", 120))
DEFAULT_BATCH_SIZE: int = int(os.environ.get("RAG_BATCH_SIZE", 128))
DEFAULT_CONTEXT_LENGTH: int = int(os.environ.get("RAG_CONTEXT_LENGTH", 4096))
DEFAULT_MAX_NEW_TOKENS: int = int(os.environ.get("RAG_MAX_NEW_TOKENS", 512))
DEFAULT_TEMPERATURE: float = float(os.environ.get("RAG_TEMPERATURE", 0.1))
DEFAULT_NUM_THREADS: int = int(os.environ.get("RAG_NUM_THREADS", 32))
DEFAULT_TOP_K: int = int(os.environ.get("RAG_TOP_K", 4))


def ensure_directories() -> None:
    """Create all required directories if they are missing."""
    for path in (DATA_DIR, CHROMA_DB_DIR, MODELS_DIR, LOG_DIR, EXAMPLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "CHROMA_DB_DIR",
    "MODELS_DIR",
    "LOG_DIR",
    "EXAMPLES_DIR",
    "DEFAULT_JSONL",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_NUM_THREADS",
    "DEFAULT_TOP_K",
    "ensure_directories",
]

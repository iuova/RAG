"""Project-wide configuration values for the local RAG pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from error_handling import ValidationError

# Загрузка переменных окружения из .env файла (если установлен python-dotenv)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv не установлен, используем только системные переменные окружения
    pass

# Base paths ---------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"
MODELS_DIR: Path = BASE_DIR / "models"
LOG_DIR: Path = BASE_DIR / "logs"
EXAMPLES_DIR: Path = BASE_DIR / "examples"

# Default resources --------------------------------------------------------
DEFAULT_JSONL: Path = DATA_DIR / "data_for_RAG.json"
DEFAULT_COLLECTION_NAME: str = os.environ.get("RAG_COLLECTION_NAME", "docs")
DEFAULT_EMBEDDING_MODEL: str = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)
DEFAULT_MODEL_FILENAME: str = os.environ.get(
    "RAG_LLM_FILENAME",
    "Qwen/Qwen2.5-7B-Instruct",
)
DEFAULT_DEVICE: str = os.environ.get("RAG_DEVICE", "auto")

# Runtime tuning -----------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))
DEFAULT_BATCH_SIZE: int = int(os.environ.get("RAG_BATCH_SIZE", "128"))
DEFAULT_EMBEDDING_BATCH_SIZE: int = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32"))
DEFAULT_CONTEXT_LENGTH: int = int(os.environ.get("RAG_CONTEXT_LENGTH", "4096"))
DEFAULT_MAX_NEW_TOKENS: int = int(os.environ.get("RAG_MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE: float = float(os.environ.get("RAG_TEMPERATURE", "0.1"))
DEFAULT_NUM_THREADS: int = int(os.environ.get("RAG_NUM_THREADS", "32"))
DEFAULT_TOP_K: int = int(os.environ.get("RAG_TOP_K", "10"))


def ensure_directories() -> None:
    """Create all required directories if they are missing."""
    for path in (DATA_DIR, CHROMA_DB_DIR, MODELS_DIR, LOG_DIR, EXAMPLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """Проверяет корректность конфигурации.

    Raises:
        ValidationError: При некорректных значениях конфигурации
    """
    if DEFAULT_CHUNK_SIZE <= 0:
        raise ValidationError(
            f"chunk_size должен быть положительным числом, получено: {DEFAULT_CHUNK_SIZE}"
        )

    if DEFAULT_CHUNK_OVERLAP < 0:
        raise ValidationError(
            f"chunk_overlap не может быть отрицательным, получено: {DEFAULT_CHUNK_OVERLAP}"
        )

    if DEFAULT_CHUNK_OVERLAP >= DEFAULT_CHUNK_SIZE:
        raise ValidationError(
            f"chunk_overlap ({DEFAULT_CHUNK_OVERLAP}) должен быть меньше "
            f"chunk_size ({DEFAULT_CHUNK_SIZE})"
        )

    if DEFAULT_BATCH_SIZE <= 0:
        raise ValidationError(
            f"batch_size должен быть положительным числом, получено: {DEFAULT_BATCH_SIZE}"
        )

    if DEFAULT_EMBEDDING_BATCH_SIZE <= 0:
        raise ValidationError(
            f"embedding_batch_size должен быть положительным числом, "
            f"получено: {DEFAULT_EMBEDDING_BATCH_SIZE}"
        )

    if DEFAULT_TOP_K <= 0:
        raise ValidationError(
            f"top_k должен быть положительным числом, получено: {DEFAULT_TOP_K}"
        )

    if DEFAULT_TEMPERATURE < 0 or DEFAULT_TEMPERATURE > 2:
        raise ValidationError(
            f"temperature должен быть в диапазоне [0, 2], получено: {DEFAULT_TEMPERATURE}"
        )

    if DEFAULT_CONTEXT_LENGTH <= 0:
        raise ValidationError(
            f"context_length должен быть положительным числом, "
            f"получено: {DEFAULT_CONTEXT_LENGTH}"
        )

    if DEFAULT_MAX_NEW_TOKENS <= 0:
        raise ValidationError(
            f"max_new_tokens должен быть положительным числом, "
            f"получено: {DEFAULT_MAX_NEW_TOKENS}"
        )

    if DEFAULT_DEVICE not in ("cpu", "cuda", "auto"):
        raise ValidationError(
            f"device должен быть 'cpu', 'cuda' или 'auto', получено: {DEFAULT_DEVICE}"
        )


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
    "DEFAULT_DEVICE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_NUM_THREADS",
    "DEFAULT_TOP_K",
    "ensure_directories",
    "validate_config",
]

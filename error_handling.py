"""Централизованная обработка ошибок для RAG проекта."""
from __future__ import annotations

import logging
import sys
from functools import wraps
from typing import Callable, TypeVar, Any

T = TypeVar("T")


class RAGError(Exception):
    """Базовый класс для ошибок RAG системы."""

    pass


class EmbeddingError(RAGError):
    """Ошибка при создании embeddings."""

    pass


class ChromaDBError(RAGError):
    """Ошибка при работе с ChromaDB."""

    pass


class ModelLoadError(RAGError):
    """Ошибка при загрузке модели."""

    pass


class ValidationError(RAGError):
    """Ошибка валидации данных."""

    pass


def handle_embedding_error(func: Callable[..., T]) -> Callable[..., T]:
    """Декоратор для обработки ошибок создания embeddings."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logging.exception("Ошибка при создании embedding")
            raise EmbeddingError(f"Не удалось создать embedding: {exc}") from exc

    return wrapper


def handle_chromadb_error(func: Callable[..., T]) -> Callable[..., T]:
    """Декоратор для обработки ошибок ChromaDB."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logging.exception("Ошибка при работе с ChromaDB")
            raise ChromaDBError(f"Ошибка ChromaDB: {exc}") from exc

    return wrapper


def safe_execute(
    func: Callable[..., T],
    error_message: str,
    error_type: type[Exception] = RAGError,
    default: T | None = None,
) -> T | None:
    """Безопасное выполнение функции с обработкой ошибок.

    Args:
        func: Функция для выполнения
        error_message: Сообщение об ошибке
        error_type: Тип исключения для логирования
        default: Значение по умолчанию при ошибке

    Returns:
        Результат функции или default при ошибке
    """
    try:
        return func()
    except Exception as exc:
        logging.exception(error_message)
        if default is not None:
            return default
        raise error_type(f"{error_message}: {exc}") from exc


def print_error_and_exit(error: Exception, exit_code: int = 1) -> None:
    """Выводит ошибку и завершает программу.

    Args:
        error: Исключение для вывода
        exit_code: Код завершения программы
    """
    error_msg = str(error)
    print(f"Ошибка: {error_msg}", file=sys.stderr)
    logging.error("Критическая ошибка: %s", error_msg)
    sys.exit(exit_code)


def format_error_for_user(error: Exception) -> str:
    """Форматирует ошибку для пользователя.

    Args:
        error: Исключение

    Returns:
        Понятное сообщение об ошибке
    """
    if isinstance(error, EmbeddingError):
        return "Ошибка при обработке запроса. Проверьте корректность входных данных."
    if isinstance(error, ChromaDBError):
        return "Ошибка при работе с базой данных. Проверьте наличие индекса."
    if isinstance(error, ModelLoadError):
        return "Ошибка при загрузке модели. Проверьте наличие модели и доступность ресурсов."
    if isinstance(error, ValidationError):
        return f"Ошибка валидации: {error}"
    return f"Произошла ошибка: {error}"


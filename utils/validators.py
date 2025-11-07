"""Валидация входных данных для RAG проекта."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from error_handling import ValidationError


def validate_question(question: str, min_length: int = 3, max_length: int = 1000) -> str:
    """Валидирует вопрос пользователя.

    Args:
        question: Текст вопроса
        min_length: Минимальная длина вопроса
        max_length: Максимальная длина вопроса

    Returns:
        Очищенный вопрос

    Raises:
        ValidationError: При некорректном вопросе
    """
    if not isinstance(question, str):
        raise ValidationError("Вопрос должен быть строкой")

    cleaned = question.strip()

    if len(cleaned) < min_length:
        raise ValidationError(
            f"Вопрос слишком короткий. Минимальная длина: {min_length} символов"
        )

    if len(cleaned) > max_length:
        raise ValidationError(
            f"Вопрос слишком длинный. Максимальная длина: {max_length} символов"
        )

    # Проверка на потенциально опасные паттерны
    dangerous_patterns = [
        r"\.\./",  # Path traversal
        r"<script",  # XSS
        r"javascript:",  # XSS
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise ValidationError("Вопрос содержит недопустимые символы")

    return cleaned


def validate_file_path(file_path: Path | str, must_exist: bool = True) -> Path:
    """Валидирует путь к файлу.

    Args:
        file_path: Путь к файлу
        must_exist: Должен ли файл существовать

    Returns:
        Валидированный Path объект

    Raises:
        ValidationError: При некорректном пути
    """
    if isinstance(file_path, str):
        path = Path(file_path)
    else:
        path = file_path

    # Проверка на path traversal
    try:
        path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        raise ValidationError("Путь выходит за пределы рабочей директории")

    if must_exist and not path.exists():
        raise ValidationError(f"Файл не найден: {path}")

    if path.is_file() and path.stat().st_size > 1024 * 1024 * 1024:  # 1GB
        raise ValidationError(f"Файл слишком большой: {path.stat().st_size / (1024**3):.2f} GB")

    return path


def validate_collection_name(name: str) -> str:
    """Валидирует имя коллекции ChromaDB.

    Args:
        name: Имя коллекции

    Returns:
        Валидированное имя

    Raises:
        ValidationError: При некорректном имени
    """
    if not isinstance(name, str):
        raise ValidationError("Имя коллекции должно быть строкой")

    cleaned = name.strip()

    if not cleaned:
        raise ValidationError("Имя коллекции не может быть пустым")

    if len(cleaned) > 100:
        raise ValidationError("Имя коллекции слишком длинное (максимум 100 символов)")

    # ChromaDB ограничения
    if not re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
        raise ValidationError(
            "Имя коллекции может содержать только буквы, цифры, дефисы и подчеркивания"
        )

    return cleaned


def validate_top_k(top_k: int, min_value: int = 1, max_value: int = 100) -> int:
    """Валидирует параметр top_k.

    Args:
        top_k: Количество документов для возврата
        min_value: Минимальное значение
        max_value: Максимальное значение

    Returns:
        Валидированное значение

    Raises:
        ValidationError: При некорректном значении
    """
    if not isinstance(top_k, int):
        raise ValidationError("top_k должен быть целым числом")

    if top_k < min_value:
        raise ValidationError(f"top_k должен быть не меньше {min_value}")

    if top_k > max_value:
        raise ValidationError(f"top_k должен быть не больше {max_value}")

    return top_k


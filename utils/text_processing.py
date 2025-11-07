"""Улучшенная обработка текста для RAG проекта."""
from __future__ import annotations

import re
from typing import List

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def split_text_smart(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    respect_sentences: bool = True,
    respect_paragraphs: bool = True,
) -> List[str]:
    """Умное разбиение текста на чанки с учетом структуры.

    Args:
        text: Исходный текст
        chunk_size: Размер чанка в символах
        chunk_overlap: Перекрытие между чанками
        respect_sentences: Учитывать границы предложений
        respect_paragraphs: Учитывать границы абзацев

    Returns:
        Список чанков текста

    Raises:
        ValueError: При некорректных параметрах
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным числом")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap не может быть отрицательным")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть меньше chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[str] = []

    # Если текст меньше размера чанка, возвращаем его целиком
    if len(normalized) <= chunk_size:
        return [normalized]

    # Разбиваем на абзацы, если нужно
    if respect_paragraphs:
        paragraphs = re.split(r"\n\s*\n", normalized)
        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Если абзац сам по себе больше чанка, разбиваем его
            if len(para) > chunk_size:
                # Сохраняем текущий чанк, если он не пустой
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Начинаем новый чанк с перекрытием
                    overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk = overlap_text + para
                    current_length = len(current_chunk)
                else:
                    # Разбиваем большой абзац на предложения
                    para_chunks = _split_large_paragraph(
                        para, chunk_size, chunk_overlap, respect_sentences
                    )
                    chunks.extend(para_chunks)
                    current_chunk = ""
                    current_length = 0
            else:
                # Проверяем, поместится ли абзац в текущий чанк
                if current_length + len(para) + 1 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                    current_length = len(current_chunk)
                else:
                    # Сохраняем текущий чанк
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # Начинаем новый с перекрытием
                    overlap_text = (
                        current_chunk[-chunk_overlap:] if chunk_overlap > 0 and current_chunk else ""
                    )
                    current_chunk = overlap_text + "\n\n" + para
                    current_length = len(current_chunk)

        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk.strip())

    else:
        # Простое разбиение без учета структуры
        chunks = _split_simple(normalized, chunk_size, chunk_overlap, respect_sentences)

    return [c for c in chunks if c]


def _split_large_paragraph(
    text: str, chunk_size: int, chunk_overlap: int, respect_sentences: bool
) -> List[str]:
    """Разбивает большой абзац на чанки.

    Args:
        text: Текст абзаца
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие
        respect_sentences: Учитывать предложения

    Returns:
        Список чанков
    """
    if respect_sentences:
        # Разбиваем на предложения
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length = len(current_chunk)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Начинаем новый с перекрытием
                overlap_text = (
                    current_chunk[-chunk_overlap:] if chunk_overlap > 0 and current_chunk else ""
                )
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    else:
        return _split_simple(text, chunk_size, chunk_overlap, False)


def _split_simple(text: str, chunk_size: int, chunk_overlap: int, respect_sentences: bool) -> List[str]:
    """Простое разбиение текста.

    Args:
        text: Исходный текст
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие
        respect_sentences: Учитывать предложения

    Returns:
        Список чанков
    """
    chunks: List[str] = []
    text_length = len(text)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if end < text_length and respect_sentences:
            # Пытаемся найти конец предложения
            sentence_end = max(
                chunk.rfind("."),
                chunk.rfind("!"),
                chunk.rfind("?"),
            )
            if sentence_end > chunk_size * 0.6:  # Не менее 60% размера чанка
                end = start + sentence_end + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= text_length:
            break

        start = max(end - chunk_overlap, 0)

    return [c for c in chunks if c]


"""Общая логика для RAG запросов."""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

from chromadb import Collection
from embedding_utils import HuggingFaceEncoder

from error_handling import (
    ChromaDBError,
    EmbeddingError,
    handle_chromadb_error,
    handle_embedding_error,
)


def clean_snippet(text: str) -> str:
    """Возвращает короткий читаемый фрагмент из чанка.

    Args:
        text: Исходный текст

    Returns:
        Очищенный фрагмент текста
    """
    normalised = re.sub(r"\s+", " ", text.strip())
    if not normalised:
        return ""
    if len(normalised) <= 280:
        return normalised
    return normalised[:277].rstrip() + "..."


def format_source(metadata: dict | None) -> str:
    """Формирует краткую метку источника из метаданных.

    Args:
        metadata: Метаданные документа

    Returns:
        Метка источника
    """
    if not metadata:
        return "Неизвестный источник"
    title = metadata.get("title")
    document_id = metadata.get("document_id")
    source = metadata.get("source")
    if title:
        return title
    if document_id and source:
        return f"{source} (ID {document_id})"
    if source:
        return source
    if document_id:
        return f"Документ {document_id}"
    return "Неизвестный источник"


@handle_embedding_error
def create_query_embedding(encoder: HuggingFaceEncoder, question: str) -> list[list[float]]:
    """Создает embedding для запроса.

    Args:
        encoder: Энкодер для создания embeddings
        question: Текст запроса

    Returns:
        Embedding запроса в формате для ChromaDB

    Raises:
        EmbeddingError: При ошибке создания embedding
    """
    try:
        query_embedding = encoder.encode_one(question).reshape(1, -1).tolist()
        return query_embedding
    except Exception as exc:
        logging.exception("Не удалось создать embedding для запроса")
        raise EmbeddingError(f"Ошибка при обработке запроса: {exc}") from exc


@handle_chromadb_error
def search_documents(
    collection: Collection,
    query_embedding: list[list[float]],
    top_k: int,
    min_relevance: float | None = None,
) -> Tuple[List[str], List[dict], List[float]]:
    """Выполняет поиск документов в коллекции.

    Args:
        collection: Коллекция ChromaDB
        query_embedding: Embedding запроса
        top_k: Количество документов для возврата
        min_relevance: Минимальная релевантность (опционально)

    Returns:
        Кортеж (documents, metadatas, distances)

    Raises:
        ChromaDBError: При ошибке поиска
    """
    # Если нужна фильтрация по релевантности, запрашиваем больше документов
    n_results = top_k * 3 if min_relevance else top_k

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logging.exception("Ошибка при выполнении запроса к Chroma")
        raise ChromaDBError(f"Не удалось выполнить поиск по коллекции: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return [], [], []

    # Фильтруем по релевантности, если указано
    if min_relevance is not None:
        filtered_data = [
            (doc, meta, dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if (1 - dist) >= min_relevance
        ]

        if not filtered_data:
            # Возвращаем все, если ничего не прошло фильтр
            return documents, metadatas, distances

        # Распаковываем отфильтрованные данные безопасно
        filtered_docs, filtered_metas, filtered_dists = zip(*filtered_data)
        return list(filtered_docs), list(filtered_metas), list(filtered_dists)

    return documents, metadatas, distances


def calculate_relevance_label(distance: float) -> str:
    """Вычисляет текстовую метку релевантности по расстоянию.

    Args:
        distance: Расстояние в векторном пространстве

    Returns:
        Метка релевантности
    """
    if distance < 0.7:
        return "высокая"
    if distance < 0.9:
        return "средняя"
    return "низкая"


def print_search_results(
    documents: List[str],
    metadatas: List[dict],
    distances: List[float],
    show_documents: bool = False,
) -> None:
    """Выводит результаты поиска.

    Args:
        documents: Найденные документы
        metadatas: Метаданные документов
        distances: Расстояния до запроса
        show_documents: Показывать ли полный текст документов
    """
    if not documents:
        print("Документов по запросу не найдено.")
        return

    print(f"Найдено {len(documents)} релевантных документов")

    if show_documents:
        print("\nНайденные документы:")
        for i, (doc, metadata, distance) in enumerate(
            zip(documents, metadatas, distances), 1
        ):
            source = format_source(metadata)
            relevance = 1 - distance
            print(f"\n{i}. {source} (relevance={relevance:.2f})")
            print(f"   {doc[:200]}..." if len(doc) > 200 else f"   {doc}")

    print(f"\nИсточники ({len(documents)}):")
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), 1):
        source = format_source(metadata)
        relevance_label = calculate_relevance_label(distance)
        relevance = 1 - distance
        print(
            f"{i}. {source} (релевантность: {relevance_label}, "
            f"relevance={relevance:.2f}, distance={distance:.4f})"
        )


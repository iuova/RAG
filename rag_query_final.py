"""Финальная версия RAG с LLM без llama.cpp - оптимальная для Windows."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from chromadb import PersistentClient
from embedding_utils import get_encoder

from config import (
    CHROMA_DB_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    LOG_DIR,
    ensure_directories,
    validate_config,
)
from error_handling import (
    ChromaDBError,
    EmbeddingError,
    format_error_for_user,
)
from rag_core import (
    clean_snippet,
    create_query_embedding,
    format_source,
    print_search_results,
    search_documents,
)


def generate_intelligent_answer(
    question: str, contexts: List[str], metadatas: List[dict]
) -> str:
    """Собирает осмысленный ответ на основе найденных фрагментов.

    Args:
        question: Текст вопроса
        contexts: Найденные контексты
        metadatas: Метаданные документов

    Returns:
        Сформированный ответ
    """
    if not contexts:
        return (
            "К сожалению, не удалось найти релевантную информацию для ответа на ваш вопрос."
        )

    parts: List[str] = []
    unique_snippets = set()

    for text, metadata in zip(contexts, metadatas):
        snippet = clean_snippet(text)
        if not snippet or snippet in unique_snippets:
            continue
        unique_snippets.add(snippet)
        source_label = format_source(metadata)
        parts.append(f"• {snippet} \n  Источник: {source_label}")

    if not parts:
        return (
            "Контексты найдены, но их не удалось преобразовать в ответ. Попробуйте уточнить вопрос."
        )

    question_label = question.strip()
    if question_label:
        intro = (
            f"По запросу «{question_label}» найдены наиболее релевантные фрагменты:"  # noqa: E501
        )
    else:
        intro = (
            "На основании найденных документов приведены наиболее релевантные фрагменты:"  # noqa: E501
        )
    return "\n".join([intro, *parts])


def run_final_rag(collection, encoder, question: str, top_k: int) -> None:
    """Выполняет финальный RAG запрос.

    Args:
        collection: Коллекция ChromaDB
        encoder: Энкодер для создания embeddings
        question: Текст вопроса
        top_k: Количество документов для возврата
    """
    print(f"Ищем информацию по запросу: '{question}'")

    # Получаем embeddings для запроса
    try:
        query_embedding = create_query_embedding(encoder, question)
    except EmbeddingError as exc:
        print(format_error_for_user(exc))
        return

    # Ищем релевантные документы
    try:
        documents, metadatas, distances = search_documents(
            collection, query_embedding, top_k
        )
    except ChromaDBError as exc:
        print(format_error_for_user(exc))
        return

    if not documents:
        print("Документов по запросу не найдено.")
        return

    # Генерируем ответ
    print(f"\nГенерируем ответ...")
    answer = generate_intelligent_answer(question, documents, metadatas)

    print(f"\nОтвет:")
    print("=" * 50)
    print(answer)
    print("=" * 50)

    # Показываем источники
    print_search_results(documents, metadatas, distances)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Финальный RAG с LLM (без llama.cpp)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Коллекция")
    parser.add_argument("--question", help="Вопрос")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Устройство для расчёта эмбеддингов (cpu, cuda, auto)",
    )
    args = parser.parse_args()

    # Валидация конфигурации
    try:
        validate_config()
    except Exception as exc:
        print(f"Ошибка конфигурации: {exc}")
        return

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_final.log"),
            logging.StreamHandler()
        ]
    )

    # Проверяем базу данных
    if not Path(CHROMA_DB_DIR).exists():
        print(f"База данных не найдена: {CHROMA_DB_DIR}")
        print("Сначала запустите: python rag_index.py data/test_data.jsonl")
        return

    # Загружаем embeddings
    print("Загружаем модель embeddings...")
    try:
        encoder = get_encoder(DEFAULT_EMBEDDING_MODEL, device=args.device)
    except Exception as exc:
        logging.exception("Не удалось инициализировать модель эмбеддингов")
        print(f"Ошибка загрузки модели эмбеддингов: {exc}")
        return

    # Подключаемся к базе
    print("Подключаемся к базе данных...")
    client = PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:
        logging.error("Collection retrieval failed: %s", exc)
        print(
            f"Коллекция '{args.collection}' не найдена. Запустите индексацию или проверьте имя коллекции."
        )
        return

    print("Система готова к работе!")

    # Обрабатываем вопрос
    if args.question:
        run_final_rag(collection, encoder, args.question, args.top_k)
        return

    # Интерактивный режим
    print("\nИнтерактивный режим. Введите вопрос (или 'exit' для выхода):")
    print("Примеры вопросов:")
    print("• Что такое докование судна?")
    print("• Как обеспечить электропитание?")
    print("• Код работы для установки трапа")
    print("• Описание пожарной безопасности")

    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                print("До свидания!")
                break
            if not question:
                continue

            run_final_rag(collection, encoder, question, args.top_k)

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {format_error_for_user(e)}")


if __name__ == "__main__":
    main()

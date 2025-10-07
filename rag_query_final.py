"""Финальная версия RAG с LLM без llama.cpp - оптимальная для Windows."""
from __future__ import annotations

import argparse
import logging
import re
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
)


def _clean_snippet(text: str) -> str:
    """Return a short, human readable snippet from a chunk."""

    normalised = re.sub(r"\s+", " ", text.strip())
    if not normalised:
        return ""
    if len(normalised) <= 280:
        return normalised
    return normalised[:277].rstrip() + "..."


def _format_source(metadata: dict | None) -> str:
    """Build a concise source label from metadata."""

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


def generate_intelligent_answer(
    question: str, contexts: List[str], metadatas: List[dict]
) -> str:
    """Собирает осмысленный ответ на основе найденных фрагментов."""

    if not contexts:
        return (
            "К сожалению, не удалось найти релевантную информацию для ответа на ваш вопрос."
        )

    parts: List[str] = []
    unique_snippets = set()

    for text, metadata in zip(contexts, metadatas):
        snippet = _clean_snippet(text)
        if not snippet or snippet in unique_snippets:
            continue
        unique_snippets.add(snippet)
        source_label = _format_source(metadata)
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
    """Выполняет финальный RAG запрос."""
    
    print(f"Ищем информацию по запросу: '{question}'")
    
    # Получаем embeddings для запроса
    try:
        query_embedding = encoder.encode_one(question).reshape(1, -1).tolist()
    except Exception as exc:
        logging.exception("Не удалось создать embedding для запроса")
        print(f"Ошибка при обработке запроса: {exc}")
        return

    # Ищем релевантные документы
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logging.exception("Ошибка при выполнении запроса к Chroma")
        print(f"Не удалось выполнить поиск по коллекции: {exc}")
        return
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not documents:
        print("Документов по запросу не найдено.")
        return
    
    print(f"Найдено {len(documents)} релевантных документов")
    
    # Генерируем ответ
    print(f"\nГенерируем ответ...")
    answer = generate_intelligent_answer(question, documents, metadatas)
    
    print(f"\nОтвет:")
    print("=" * 50)
    print(answer)
    print("=" * 50)
    
    # Показываем источники
    print(f"\nИсточники ({len(documents)}):")
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        relevance = "высокая" if distance < 0.7 else "средняя" if distance < 0.9 else "низкая"
        print(f"{i}. {source} (релевантность: {relevance}, distance={distance:.4f})")


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
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

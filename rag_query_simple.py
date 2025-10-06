"""Простая версия RAG запросов без llama-cpp для тестирования."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from chromadb import PersistentClient

from embedding_utils import HuggingFaceEncoder

from config import (
    CHROMA_DB_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    LOG_DIR,
    ensure_directories,
)


def format_simple_response(question: str, contexts: List[str]) -> str:
    """Простое форматирование ответа без LLM."""
    context_block = "\n\n".join(f"Источник {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts))
    return f"""
Вопрос: {question}

Найденные релевантные документы:
{context_block}

[Примечание: Для полного ответа требуется установка llama-cpp-python]
"""


def run_simple_query(
    collection, encoder: HuggingFaceEncoder, question: str, top_k: int
) -> None:
    """Выполняет простой поиск без генерации ответа."""

    query_embedding = encoder.encode_one(question).reshape(1, -1).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        print("Документов по запросу не найдено. Убедитесь, что индекс создан.")
        return

    response = format_simple_response(question, documents)
    print(response)

    print("\n" + "=" * 50)
    print("ИСТОЧНИКИ:")
    distances = results.get("distances", [[]])[0]
    for i, metadata in enumerate(metadatas, 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        distance = distances[i - 1] if i - 1 < len(distances) else float("nan")
        print(f"{i}. {source} (distance={distance:.4f})")


def main():
    """Основная функция для запуска простых RAG запросов."""
    parser = argparse.ArgumentParser(description="Простой RAG без LLM")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов для поиска")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Имя коллекции Chroma")
    args = parser.parse_args()

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_simple.log"),
            logging.StreamHandler()
        ]
    )

    # Проверяем существование базы данных
    if not Path(CHROMA_DB_DIR).exists():
        print(f"База данных не найдена: {CHROMA_DB_DIR}")
        print("Сначала запустите: python rag_index.py examples/example_documents.jsonl --reset")
        return

    # Загружаем embeddings и векторную базу
    print("Загружаем модель embeddings...")
    encoder = HuggingFaceEncoder(DEFAULT_EMBEDDING_MODEL, device="cpu")

    print("Подключаемся к векторной базе...")
    client = PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"Коллекция '{args.collection}' не найдена.")
        return

    print("Система готова к работе!")
    print("Введите ваш вопрос (или 'exit' для выхода):")
    
    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                break
            if not question:
                continue
                
            print(f"\nИщем релевантные документы для: '{question}'")
            run_simple_query(collection, encoder, question, args.top_k)
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

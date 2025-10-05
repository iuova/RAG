"""Простая версия RAG запросов без llama-cpp для тестирования."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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


def run_simple_query(retriever: Chroma, question: str, top_k: int) -> None:
    """Выполняет простой поиск без генерации ответа."""
    docs = retriever.similarity_search(question, k=top_k)
    if not docs:
        print("Документов по запросу не найдено. Убедитесь, что индекс создан.")
        return

    contexts = [doc.page_content for doc in docs]
    response = format_simple_response(question, contexts)
    print(response)
    
    # Показываем источники
    print("\n" + "="*50)
    print("ИСТОЧНИКИ:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Неизвестно')
        print(f"{i}. {source}")


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
    embeddings = HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    print("Подключаемся к векторной базе...")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings,
        collection_name=args.collection
    )

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
            run_simple_query(vector_store, question, args.top_k)
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

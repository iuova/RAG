"""Гибридный RAG: поиск + простая генерация ответов."""
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


def generate_simple_answer(question: str, contexts: List[str]) -> str:
    """Генерирует простой ответ на основе найденных документов."""
    
    # Анализируем вопрос
    question_lower = question.lower()
    
    # Определяем тип вопроса
    if any(word in question_lower for word in ["что", "как", "где", "когда", "почему"]):
        question_type = "описательный"
    elif any(word in question_lower for word in ["код", "номер", "шифр"]):
        question_type = "код"
    elif any(word in question_lower for word in ["описание", "детали", "подробности"]):
        question_type = "описание"
    else:
        question_type = "общий"
    
    # Формируем ответ на основе найденных документов
    answer_parts = []
    
    for i, context in enumerate(contexts, 1):
        # Извлекаем информацию из контекста
        if "Код работы:" in context:
            code_part = context.split("Код работы:")[1].split("|")[0].strip()
            if question_type == "код":
                answer_parts.append(f"Код работы: {code_part}")
        
        if "Пункт:" in context:
            point_part = context.split("Пункт:")[1].split("|")[0].strip()
            if question_type in ["описательный", "общий"]:
                answer_parts.append(f"Пункт ремонтной ведомости: {point_part}")
        
        if "Описание:" in context:
            desc_part = context.split("Описание:")[1].strip()
            if question_type in ["описание", "описательный"]:
                answer_parts.append(f"Описание: {desc_part}")
    
    # Формируем итоговый ответ
    if answer_parts:
        answer = "На основе найденных документов:\n\n" + "\n\n".join(answer_parts)
    else:
        answer = "Найдена следующая информация:\n\n" + "\n\n".join(contexts)
    
    return answer


def run_hybrid_rag(
    collection, encoder: HuggingFaceEncoder, question: str, top_k: int
) -> None:
    """Выполняет гибридный RAG запрос."""
    
    # Получаем embeddings для запроса
    query_embedding = encoder.encode_one(question).reshape(1, -1).tolist()
    
    # Ищем релевантные документы
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not documents:
        print("Документов по запросу не найдено.")
        return
    
    print(f"Найдено {len(documents)} релевантных документов")
    
    # Показываем найденные документы
    print("\n=== Найденные документы ===")
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        print(f"\n{i}. {source} (distance={distance:.4f})")
        print(f"   {doc[:200]}...")
    
    # Генерируем ответ
    print(f"\n=== Сгенерированный ответ ===")
    answer = generate_simple_answer(question, documents)
    print(answer)
    
    # Показываем источники
    print(f"\n=== Источники ===")
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        print(f"{i}. {source} (distance={distance:.4f})")


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Гибридный RAG")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Коллекция")
    parser.add_argument("--question", help="Вопрос")
    args = parser.parse_args()

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_hybrid.log"),
            logging.StreamHandler()
        ]
    )

    # Проверяем базу данных
    if not Path(CHROMA_DB_DIR).exists():
        print(f"База данных не найдена: {CHROMA_DB_DIR}")
        return

    # Загружаем embeddings
    print("Загружаем embeddings...")
    encoder = HuggingFaceEncoder(DEFAULT_EMBEDDING_MODEL, device="cpu")

    # Подключаемся к базе
    print("Подключаемся к базе данных...")
    client = PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"Коллекция '{args.collection}' не найдена.")
        return

    print("Система готова!")
    
    # Обрабатываем вопрос
    if args.question:
        run_hybrid_rag(collection, encoder, args.question, args.top_k)
        return
    
    # Интерактивный режим
    print("Введите вопрос (или 'exit'):")
    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                break
            if not question:
                continue
                
            run_hybrid_rag(collection, encoder, question, args.top_k)
            
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

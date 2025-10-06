"""Простой RAG с LLM генерацией без сложных настроек."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from transformers import pipeline
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


def format_simple_prompt(question: str, contexts: List[str]) -> str:
    """Простое форматирование промпта."""
    context_text = " ".join(contexts)
    return f"Контекст: {context_text}\n\nВопрос: {question}\nОтвет:"


def run_simple_llm_rag(
    generator, collection, encoder: HuggingFaceEncoder, question: str, top_k: int
) -> None:
    """Выполняет простой RAG запрос с LLM."""
    
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
    
    # Формируем простой промпт
    prompt = format_simple_prompt(question, documents)
    
    print(f"\n=== Генерация ответа ===")
    print("Генерируем ответ...")
    
    try:
        # Генерируем ответ
        response = generator(
            prompt,
            max_new_tokens=100,  # Используем max_new_tokens вместо max_length
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        
        generated_text = response[0]['generated_text']
        # Извлекаем только ответ
        if "Ответ:" in generated_text:
            answer = generated_text.split("Ответ:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        print(f"\nОтвет: {answer}")
        
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        print("Показываем найденные документы как ответ:")
        for i, doc in enumerate(documents, 1):
            print(f"\nДокумент {i}: {doc}")


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Простой RAG с LLM")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Коллекция")
    parser.add_argument("--model", default="distilgpt2", help="Модель для генерации")
    parser.add_argument("--question", help="Вопрос")
    args = parser.parse_args()

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_simple_llm.log"),
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

    # Загружаем модель
    print(f"Загружаем модель: {args.model}")
    try:
        generator = pipeline(
            "text-generation",
            model=args.model,
            device=-1,  # CPU
            return_full_text=False
        )
        
        # Устанавливаем pad_token
        if generator.tokenizer.pad_token is None:
            generator.tokenizer.pad_token = generator.tokenizer.eos_token
            
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    print("Система готова!")
    
    # Обрабатываем вопрос
    if args.question:
        run_simple_llm_rag(generator, collection, encoder, args.question, args.top_k)
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
                
            run_simple_llm_rag(generator, collection, encoder, question, args.top_k)
            
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

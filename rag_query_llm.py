"""Улучшенный RAG с LLM для генерации ответов без llama.cpp."""
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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


def format_prompt(question: str, contexts: List[str]) -> str:
    """Форматирует промпт для модели."""
    context_block = "\n\n".join(f"Источник {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts))
    return f"""Ты — экспертный технический ассистент по ремонтным работам судов. Отвечай на русском языке.
Используй только факты из приведённого контекста. Если информации недостаточно,
сообщи об этом честно.

Контекст:
{context_block}

Вопрос: {question}
Ответ:"""


def run_llm_rag_query(
    generator, collection, encoder: HuggingFaceEncoder, question: str, top_k: int
) -> None:
    """Выполняет RAG запрос с LLM генерацией."""
    
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
        print("Документов по запросу не найдено. Убедитесь, что индекс создан.")
        return
    
    # Формируем промпт
    prompt = format_prompt(question, documents)
    
    print(f"Генерируем ответ на основе {len(documents)} документов...")
    
    try:
        # Генерируем ответ
        response = generator(
            prompt,
            max_length=len(prompt.split()) + 150,  # Ограничиваем длину ответа
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
        )
        
        generated_text = response[0]['generated_text']
        # Извлекаем только ответ (после "Ответ:")
        if "Ответ:" in generated_text:
            answer = generated_text.split("Ответ:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        print(f"\n=== Ответ ===")
        print(answer)
        
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        print("Показываем найденные документы:")
        for i, doc in enumerate(documents, 1):
            print(f"\nДокумент {i}:")
            print(doc)
    
    # Показываем источники
    print(f"\n=== Источники ({len(documents)}) ===")
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        print(f"{i}. {source} (distance={distance:.4f})")


def main():
    """Основная функция для запуска RAG с LLM."""
    parser = argparse.ArgumentParser(description="RAG с LLM генерацией")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов для поиска")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Имя коллекции Chroma")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Модель для генерации")
    parser.add_argument("--question", help="Вопрос для обработки (необязательно)")
    args = parser.parse_args()

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_llm.log"),
            logging.StreamHandler()
        ]
    )

    # Проверяем существование базы данных
    if not Path(CHROMA_DB_DIR).exists():
        print(f"База данных не найдена: {CHROMA_DB_DIR}")
        print("Сначала запустите: python rag_index.py data/test_data.jsonl")
        return

    # Загружаем embeddings
    print("Загружаем модель embeddings...")
    encoder = HuggingFaceEncoder(DEFAULT_EMBEDDING_MODEL, device="cpu")

    # Подключаемся к векторной базе
    print("Подключаемся к векторной базе...")
    client = PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"Коллекция '{args.collection}' не найдена.")
        return

    # Загружаем модель для генерации
    print(f"Загружаем модель генерации: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        
        # Создаем pipeline для генерации текста
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
            return_full_text=False
        )
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Попробуйте другую модель или проверьте подключение к интернету")
        return

    print("Система готова к работе!")
    
    # Если задан вопрос через аргумент
    if args.question:
        print(f"Обрабатываем вопрос: {args.question}")
        run_llm_rag_query(generator, collection, encoder, args.question, args.top_k)
        return
    
    # Интерактивный режим
    print("Введите ваш вопрос (или 'exit' для выхода):")
    
    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                break
            if not question:
                continue
                
            run_llm_rag_query(generator, collection, encoder, question, args.top_k)
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

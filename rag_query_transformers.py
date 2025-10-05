"""RAG с использованием transformers для генерации ответов."""
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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


def format_prompt(question: str, contexts: List[str]) -> str:
    """Форматирует промпт для модели."""
    context_block = "\n\n".join(f"Источник {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts))
    return f"""Ты — экспертный технический ассистент. Отвечай на русском языке.
Используй только факты из приведённого контекста. Если информации недостаточно,
сообщи об этом честно.

Контекст:
{context_block}

Вопрос: {question}
Ответ:"""


def run_query_with_transformers(generator, retriever: Chroma, question: str, top_k: int) -> None:
    """Выполняет RAG запрос с использованием transformers."""
    docs = retriever.similarity_search(question, k=top_k)
    if not docs:
        print("Документов по запросу не найдено. Убедитесь, что индекс создан.")
        return

    contexts = [doc.page_content for doc in docs]
    prompt = format_prompt(question, contexts)
    
    print(f"Генерируем ответ...")
    
    # Генерируем ответ
    response = generator(
        prompt,
        max_length=len(prompt.split()) + 100,  # Ограничиваем длину ответа
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    generated_text = response[0]['generated_text']
    # Извлекаем только ответ (после "Ответ:")
    answer = generated_text.split("Ответ:")[-1].strip()
    
    print(f"\nОтвет: {answer}")
    
    # Показываем источники
    print("\n" + "="*50)
    print("ИСТОЧНИКИ:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Неизвестно')
        print(f"{i}. {source}")


def main():
    """Основная функция для запуска RAG с transformers."""
    parser = argparse.ArgumentParser(description="RAG с transformers")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов для поиска")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Имя коллекции Chroma")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Модель для генерации")
    args = parser.parse_args()

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_transformers.log"),
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
    print("Введите ваш вопрос (или 'exit' для выхода):")
    
    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                break
            if not question:
                continue
                
            print(f"\nИщем релевантные документы для: '{question}'")
            run_query_with_transformers(generator, vector_store, question, args.top_k)
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

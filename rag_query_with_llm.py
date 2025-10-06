"""RAG с настоящей LLM для генерации ответов на русском языке."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

warnings.filterwarnings('ignore')


def format_prompt_for_llm(question: str, contexts: List[str]) -> str:
    """Форматирует промпт для Qwen2.5 модели."""
    context_block = "\n\n".join(f"Документ {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts))
    
    prompt = f"""<|im_start|>system
Ты - профессиональный ассистент по ремонтным работам судов. Отвечай на русском языке.
Используй ТОЛЬКО информацию из предоставленных документов. Если информации недостаточно, скажи об этом честно.
Будь кратким и информативным.<|im_end|>

<|im_start|>user
Контекст (документы):
{context_block}

Вопрос: {question}<|im_end|>

<|im_start|>assistant
"""
    return prompt


def generate_answer_with_llm(
    model, tokenizer, question: str, contexts: List[str], device: str
) -> str:
    """Генерирует ответ используя LLM модель."""
    
    # Формируем промпт
    prompt = format_prompt_for_llm(question, contexts)
    
    # Токенизируем
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Декодируем ответ
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только ответ ассистента
    if "<|im_start|>assistant" in generated_text:
        answer = generated_text.split("<|im_start|>assistant")[-1].strip()
    else:
        answer = generated_text[len(prompt):].strip()
    
    return answer


def run_llm_rag_query(
    model, tokenizer, collection, encoder: HuggingFaceEncoder, 
    question: str, top_k: int, device: str
) -> None:
    """Выполняет RAG запрос с LLM генерацией."""
    
    print(f"Ищем релевантные документы для: '{question}'")
    
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
        print(f"\n{i}. {source} (relevance={1-distance:.2f})")
        print(f"   {doc[:150]}...")
    
    # Генерируем ответ с помощью LLM
    print(f"\nГенерируем ответ с помощью LLM...")
    try:
        answer = generate_answer_with_llm(model, tokenizer, question, documents, device)
        
        print(f"\n=== Ответ LLM ===")
        print(answer)
        
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        print("Показываем найденные документы:")
        for i, doc in enumerate(documents, 1):
            print(f"\n{i}. {doc}")
    
    # Показываем источники
    print(f"\n=== Источники ({len(documents)}) ===")
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        print(f"{i}. {source} (distance={distance:.4f})")


def load_llm_model(model_name: str, device: str):
    """Загружает LLM модель."""
    print(f"Загружаем LLM модель: {model_name}")
    print("Это может занять несколько минут...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Модель успешно загружена!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="RAG с LLM генерацией")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Коллекция")
    parser.add_argument(
        "--model", 
        # default="Qwen/Qwen2.5-7B-Instruct",
        default=r".\models\Qwen2.5-7B-Instruct",
        help="LLM модель (по умолчанию: Qwen2.5-7B-Instruct для русского языка)"
    )
    parser.add_argument("--question", help="Вопрос")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Устройство")
    args = parser.parse_args()

    # Настройка логирования
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "rag_query_with_llm.log"),
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
    encoder = HuggingFaceEncoder(DEFAULT_EMBEDDING_MODEL, device="cpu")

    # Подключаемся к базе
    print("Подключаемся к базе данных...")
    client = PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"Коллекция '{args.collection}' не найдена.")
        return

    # Загружаем LLM
    model, tokenizer = load_llm_model(args.model, args.device)
    if model is None or tokenizer is None:
        print("Не удалось загрузить LLM модель.")
        return

    print("Система готова к работе!")
    
    # Обрабатываем вопрос
    if args.question:
        run_llm_rag_query(model, tokenizer, collection, encoder, args.question, args.top_k, args.device)
        return
    
    # Интерактивный режим
    print("\nИнтерактивный режим. Введите вопрос (или 'exit' для выхода):")
    print("Примеры вопросов:")
    print("• Что такое докование судна?")
    print("• Как обеспечить электропитание?")
    print("• Какие работы нужны для пожарной безопасности?")
    
    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                print("До свидания!")
                break
            if not question:
                continue
                
            run_llm_rag_query(model, tokenizer, collection, encoder, question, args.top_k, args.device)
            
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
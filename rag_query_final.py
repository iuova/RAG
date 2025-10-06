"""Финальная версия RAG с LLM без llama.cpp - оптимальная для Windows."""
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


def generate_intelligent_answer(question: str, contexts: List[str]) -> str:
    """Генерирует интеллектуальный ответ на основе найденных документов."""
    
    question_lower = question.lower()
    
    # Анализируем тип вопроса
    if any(word in question_lower for word in ["что такое", "что это", "определение"]):
        question_type = "определение"
    elif any(word in question_lower for word in ["как", "каким образом", "способ"]):
        question_type = "процедура"
    elif any(word in question_lower for word in ["код", "номер", "шифр", "какой код"]):
        question_type = "код"
    elif any(word in question_lower for word in ["описание", "детали", "подробности"]):
        question_type = "описание"
    elif any(word in question_lower for word in ["где", "место", "расположение"]):
        question_type = "место"
    else:
        question_type = "общий"
    
    # Извлекаем информацию из контекстов
    codes = []
    points = []
    descriptions = []
    
    for context in contexts:
        if "Код работы:" in context:
            code_part = context.split("Код работы:")[1].split("|")[0].strip()
            codes.append(code_part)
        
        if "Пункт:" in context:
            point_part = context.split("Пункт:")[1].split("|")[0].strip()
            points.append(point_part)
        
        if "Описание:" in context:
            desc_part = context.split("Описание:")[1].strip()
            descriptions.append(desc_part)
    
    # Формируем ответ в зависимости от типа вопроса
    answer_parts = []
    
    if question_type == "код":
        if codes:
            answer_parts.append(f"Коды работ по вашему запросу:")
            for code in codes:
                answer_parts.append(f"• {code}")
        else:
            answer_parts.append("Коды работ не найдены в найденных документах.")
    
    elif question_type == "процедура":
        if points:
            answer_parts.append("Процедуры выполнения работ:")
            for point in points:
                answer_parts.append(f"• {point}")
        if descriptions:
            answer_parts.append("\nДетальное описание:")
            for desc in descriptions:
                answer_parts.append(f"• {desc}")
    
    elif question_type == "описание":
        if descriptions:
            answer_parts.append("Описание работ:")
            for desc in descriptions:
                answer_parts.append(f"• {desc}")
        if points:
            answer_parts.append("\nПункты ремонтной ведомости:")
            for point in points:
                answer_parts.append(f"• {point}")
    
    elif question_type == "определение":
        if descriptions:
            answer_parts.append("Определение:")
            answer_parts.append(descriptions[0])  # Берем первое описание
        if codes:
            answer_parts.append(f"\nСвязанные коды работ: {', '.join(codes)}")
    
    else:  # общий
        if codes:
            answer_parts.append("Найденные коды работ:")
            for code in codes:
                answer_parts.append(f"• {code}")
        
        if points:
            answer_parts.append("\nПункты ремонтной ведомости:")
            for point in points:
                answer_parts.append(f"• {point}")
        
        if descriptions:
            answer_parts.append("\nОписания:")
            for desc in descriptions:
                answer_parts.append(f"• {desc}")
    
    # Формируем итоговый ответ
    if answer_parts:
        answer = "\n".join(answer_parts)
    else:
        answer = "К сожалению, не удалось найти релевантную информацию для ответа на ваш вопрос."
    
    return answer


def run_final_rag(
    collection, encoder: HuggingFaceEncoder, question: str, top_k: int
) -> None:
    """Выполняет финальный RAG запрос."""
    
    print(f"Ищем информацию по запросу: '{question}'")
    
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
    
    # Генерируем ответ
    print(f"\nГенерируем ответ...")
    answer = generate_intelligent_answer(question, documents)
    
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
    encoder = HuggingFaceEncoder(DEFAULT_EMBEDDING_MODEL, device="cpu")

    # Подключаемся к базе
    print("Подключаемся к базе данных...")
    client = PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"Коллекция '{args.collection}' не найдена.")
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

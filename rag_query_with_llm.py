"""RAG с настоящей LLM для генерации ответов на русском языке."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    ModelLoadError,
    format_error_for_user,
)
from rag_core import (
    create_query_embedding,
    print_search_results,
    search_documents,
)

warnings.filterwarnings('ignore')

# Глобальные переменные для кэширования модели
_cached_model: Optional[AutoModelForCausalLM] = None
_cached_tokenizer: Optional[AutoTokenizer] = None
_cached_model_name: Optional[str] = None
_cached_device: Optional[str] = None


def _resolve_device(device: str) -> str:
    """Normalize device strings with auto-detection.

    Args:
        device: Строка устройства

    Returns:
        Нормализованное устройство
    """
    lowered = device.lower()
    if lowered == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if lowered == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA запрошена, но недоступна. Используем CPU.")
        return "cpu"
    return lowered


def format_prompt_for_llm(question: str, contexts: List[str]) -> str:
    """Форматирует промпт для Qwen2.5 модели.

    Args:
        question: Текст вопроса
        contexts: Найденные контексты

    Returns:
        Отформатированный промпт
    """
    context_block = "\n\n".join(f"Документ {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts))

    prompt = f"""<|im_start|>system
You are a professional assistant for ship repair work. Answer ONLY in Russian language.
Use ONLY information from the provided documents. Carefully analyze the information in the provided documents.
If information is insufficient, say so honestly. Be brief and informative. The answer should contain only the codes of found works.<|im_end|>

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
    """Генерирует ответ используя LLM модель.

    Args:
        model: LLM модель
        tokenizer: Токенизатор
        question: Текст вопроса
        contexts: Найденные контексты
        device: Устройство для генерации

    Returns:
        Сгенерированный ответ
    """
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

    # Декодируем полный ответ (весь сгенерированный текст)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Логируем для отладки (первые 500 символов)
    logging.debug(f"Сгенерированный текст (первые 500 символов): {generated_text[:500]}")

    # Извлекаем полный ответ ассистента (без обрезки)
    if "<|im_start|>assistant" in generated_text:
        # Берем все после тега assistant, включая возможные теги
        answer = generated_text.split("<|im_start|>assistant")[-1]
        # Убираем только закрывающий тег, но оставляем весь текст до него
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0]
        answer = answer.strip()
    else:
        # Если не нашли тег, берем все после промпта
        answer = generated_text[len(prompt):].strip()

    # Проверяем, что ответ не пустой
    if not answer:
        logging.warning("Модель вернула пустой ответ. Сгенерированный текст:")
        logging.warning(f"Полный текст: {generated_text}")
        # Пытаемся извлечь ответ другим способом
        if prompt in generated_text:
            answer = generated_text.split(prompt, 1)[-1].strip()
        else:
            answer = generated_text.strip()

    return answer if answer else "Модель не смогла сгенерировать ответ."


def run_llm_rag_query(
    model,
    tokenizer,
    collection,
    encoder,
    question: str,
    top_k: int,
    device: str,
) -> None:
    """Выполняет RAG запрос с LLM генерацией.

    Args:
        model: LLM модель
        tokenizer: Токенизатор
        collection: Коллекция ChromaDB
        encoder: Энкодер для создания embeddings
        question: Текст вопроса
        top_k: Количество документов для возврата
        device: Устройство для генерации
    """
    print(f"Ищем релевантные документы для: '{question}'")

    runtime_device = _resolve_device(device)

    # Получаем embeddings для запроса
    try:
        query_embedding = create_query_embedding(encoder, question)
    except EmbeddingError as exc:
        print(format_error_for_user(exc))
        return

    # Фильтруем документы по relevance > 0.6
    MIN_RELEVANCE = 0.6

    # Ищем релевантные документы
    try:
        documents, metadatas, distances = search_documents(
            collection, query_embedding, top_k, min_relevance=MIN_RELEVANCE
        )
    except ChromaDBError as exc:
        print(format_error_for_user(exc))
        return

    if not documents:
        print("Документов по запросу не найдено.")
        return

    # Используем все документы с relevance > MIN_RELEVANCE для генерации ответа
    print(f"Найдено {len(documents)} релевантных документов (relevance > {MIN_RELEVANCE})")
    print(f"Все найденные документы будут использованы для генерации ответа")

    # Показываем найденные документы
    print("\nНайдены строки:")
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
        source = metadata.get("source", "Неизвестно") if metadata else "Неизвестно"
        relevance = 1 - distance
        print(f"\n{i}. {source} (relevance={relevance:.2f})")
        print(f"   {doc[:200]}..." if len(doc) > 200 else f"   {doc}")

    # Генерируем ответ с помощью LLM
    print(f"\nГенерируем ответ с помощью LLM...")
    try:
        answer = generate_answer_with_llm(
            model, tokenizer, question, documents, runtime_device
        )

        if not answer or answer.strip() == "":
            print("\n⚠️ ВНИМАНИЕ: Модель вернула пустой ответ!")
            print("Проверьте логи для подробной информации.")
            logging.warning("Пустой ответ от модели")
        else:
            print(f"\n=== Полный ответ LLM ===")
            try:
                print(answer)
            except UnicodeEncodeError:
                # Для Windows терминала с cp1251
                print(answer.encode('cp1251', errors='ignore').decode('cp1251'))

    except Exception as e:
        logging.exception("Ошибка при генерации ответа")
        print(f"Ошибка генерации: {format_error_for_user(e)}")
        print("Показываем найденные документы:")
        for i, doc in enumerate(documents, 1):
            print(f"\n{i}. {doc[:200]}..." if len(doc) > 200 else f"\n{i}. {doc}")

    # Показываем источники
    print_search_results(documents, metadatas, distances)


def load_llm_model(model_name: str, device: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Загружает LLM модель с кэшированием.

    Args:
        model_name: Имя модели
        device: Устройство для загрузки

    Returns:
        Кортеж (model, tokenizer) или (None, None) при ошибке
    """
    global _cached_model, _cached_tokenizer, _cached_model_name, _cached_device

    resolved_device = _resolve_device(device)

    # Проверяем, загружена ли уже модель с теми же параметрами
    if (_cached_model is not None and
        _cached_tokenizer is not None and
        _cached_model_name == model_name and
        _cached_device == resolved_device):
        print(f"Модель {model_name} уже загружена в память, используем кэш.")
        return _cached_model, _cached_tokenizer

    print(f"Загружаем LLM модель: {model_name}")
    print("Загружаем локальную модель (может занять время на инициализацию)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if resolved_device == "cuda" else torch.float32,
            device_map="auto" if resolved_device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if resolved_device == "cpu":
            model = model.to(resolved_device)

        model.eval()

        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Сохраняем в кэш
        _cached_model = model
        _cached_tokenizer = tokenizer
        _cached_model_name = model_name
        _cached_device = resolved_device

        print("Модель успешно загружена и сохранена в кэш!")
        return model, tokenizer

    except Exception as e:
        logging.exception("Ошибка загрузки модели")
        raise ModelLoadError(f"Не удалось загрузить модель: {e}") from e


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="RAG с LLM генерацией")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество документов")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Коллекция")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="LLM модель (по умолчанию: Qwen2.5-7B-Instruct для русского языка)",
    )
    parser.add_argument("--question", help="Вопрос")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "auto"],
        help="Устройство для генерации и эмбеддингов",
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
    runtime_device = _resolve_device(args.device)

    print("Загружаем модель embeddings...")
    try:
        encoder = get_encoder(DEFAULT_EMBEDDING_MODEL, device=runtime_device)
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

    # Загружаем LLM (с проверкой кэша)
    try:
        model, tokenizer = load_llm_model(args.model, runtime_device)
    except ModelLoadError as exc:
        print(format_error_for_user(exc))
        return

    if model is None or tokenizer is None:
        print("Не удалось загрузить LLM модель.")
        return

    print("Система готова к работе!")

    # Обрабатываем вопрос
    if args.question:
        run_llm_rag_query(
            model,
            tokenizer,
            collection,
            encoder,
            args.question,
            args.top_k,
            runtime_device,
        )
        return

    # Интерактивный режим
    print("\nИнтерактивный режим. Введите вопрос (или 'exit' для выхода):")

    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ['exit', 'quit', 'выход']:
                print("До свидания!")
                break
            if not question:
                continue

            run_llm_rag_query(
                model,
                tokenizer,
                collection,
                encoder,
                question,
                args.top_k,
                runtime_device,
            )

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {format_error_for_user(e)}")


if __name__ == "__main__":
    main()

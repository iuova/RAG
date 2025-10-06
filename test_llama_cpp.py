"""Тестирование llama.cpp для RAG системы."""
import sys
from pathlib import Path

def test_llama_cpp_availability():
    """Проверяет доступность llama.cpp."""
    print("=== Проверка llama.cpp ===")
    
    try:
        import llama_cpp
        print("llama-cpp-python установлен")
        print(f"Версия: {llama_cpp.__version__}")
    except ImportError:
        print("llama-cpp-python не установлен")
        print("Установите: pip install llama-cpp-python")
        return False
    
    return True

def test_model_availability():
    """Проверяет наличие GGUF модели."""
    print("\n=== Проверка модели ===")
    
    from config import MODELS_DIR, DEFAULT_MODEL_FILENAME
    
    model_path = MODELS_DIR / DEFAULT_MODEL_FILENAME
    print(f"Ожидаемый путь к модели: {model_path}")
    
    if model_path.exists():
        print(f"Модель найдена: {model_path}")
        print(f"Размер файла: {model_path.stat().st_size / (1024*1024*1024):.2f} GB")
        return True
    else:
        print(f"Модель не найдена: {model_path}")
        print("Нужен GGUF файл для llama.cpp")
        
        # Проверим, есть ли другие GGUF файлы
        gguf_files = list(MODELS_DIR.glob("*.gguf"))
        if gguf_files:
            print(f"Найдены GGUF файлы: {[f.name for f in gguf_files]}")
        else:
            print("GGUF файлы не найдены")
        
        return False

def test_llama_cpp_loading():
    """Тестирует загрузку модели в llama.cpp."""
    print("\n=== Тестирование загрузки модели ===")
    
    try:
        from llama_cpp import Llama
        from config import MODELS_DIR, DEFAULT_MODEL_FILENAME, DEFAULT_NUM_THREADS, DEFAULT_CONTEXT_LENGTH
        
        model_path = MODELS_DIR / DEFAULT_MODEL_FILENAME
        
        if not model_path.exists():
            print("❌ Модель не найдена, пропускаем тест")
            return False
        
        print("Загружаем модель...")
        llm = Llama(
            model_path=str(model_path),
            n_threads=DEFAULT_NUM_THREADS,
            n_ctx=DEFAULT_CONTEXT_LENGTH,
            verbose=False,
        )
        
        print("Модель успешно загружена")
        
        # Простой тест генерации
        print("Тестируем генерацию...")
        response = llm.create_completion(
            prompt="Привет! Как дела?",
            max_tokens=50,
            temperature=0.7,
        )
        
        generated_text = response["choices"][0]["text"].strip()
        print(f"Генерация работает: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False

def test_rag_with_llama():
    """Тестирует полный RAG с llama.cpp."""
    print("\n=== Тестирование RAG с llama.cpp ===")
    
    try:
        # Проверим, есть ли база данных
        from config import CHROMA_DB_DIR
        if not CHROMA_DB_DIR.exists():
            print("База данных не найдена. Сначала запустите индексацию.")
            return False
        
        print("База данных найдена")
        
        # Попробуем запустить rag_query.py
        import subprocess
        
        print("Запускаем rag_query.py...")
        result = subprocess.run([
            sys.executable, "rag_query.py", 
            "--top-k", "2"
        ], 
        input="докование судна\nexit\n", 
        text=True, 
        capture_output=True, 
        timeout=60
        )
        
        if result.returncode == 0:
            print("RAG с llama.cpp работает")
            print("Результат:")
            output_lines = result.stdout.split('\n')[:10]
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"Ошибка RAG: {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Таймаут RAG теста")
        return False
    except Exception as e:
        print(f"Ошибка RAG теста: {e}")
        return False

def main():
    """Основная функция тестирования llama.cpp."""
    print("Тестирование llama.cpp для RAG системы")
    print("=" * 50)
    
    # Проверяем доступность llama.cpp
    if not test_llama_cpp_availability():
        return
    
    # Проверяем наличие модели
    if not test_model_availability():
        print("\n💡 Рекомендации:")
        print("1. Скачайте GGUF модель Qwen2.5-7B-Instruct")
        print("2. Поместите файл в папку models/")
        print("3. Или используйте rag_query_transformers.py для работы с transformers")
        return
    
    # Тестируем загрузку модели
    if not test_llama_cpp_loading():
        return
    
    # Тестируем полный RAG
    test_rag_with_llama()
    
    print("\n" + "=" * 50)
    print("Тестирование llama.cpp завершено")

if __name__ == "__main__":
    main()

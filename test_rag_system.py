"""Тестирование RAG системы на данных из папки data."""
import subprocess
import sys
from pathlib import Path

def test_simple_search():
    """Тестирует простой поиск без генерации."""
    print("=== Тестирование простого поиска ===")
    
    test_queries = [
        "докование судна",
        "электропитание",
        "пожарная безопасность",
        "трап сходня",
        "швартовые операции"
    ]
    
    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        try:
            # Запускаем простой поиск
            result = subprocess.run([
                sys.executable, "rag_query_simple.py", 
                "--top-k", "3"
            ], 
            input=f"{query}\nexit\n", 
            text=True, 
            capture_output=True, 
            timeout=30
            )
            
            if result.returncode == 0:
                print("Поиск выполнен успешно")
                # Показываем первые несколько строк результата
                output_lines = result.stdout.split('\n')[:10]
                for line in output_lines:
                    if line.strip():
                        print(f"  {line}")
            else:
                print(f"Ошибка поиска: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Таймаут поиска")
        except Exception as e:
            print(f"Ошибка: {e}")

def test_transformers_search():
    """Тестирует поиск с transformers."""
    print("\n=== Тестирование поиска с transformers ===")
    
    test_query = "докование судна"
    print(f"Запрос: '{test_query}'")
    
    try:
        result = subprocess.run([
            sys.executable, "rag_query_transformers.py", 
            "--top-k", "2"
        ], 
        input=f"{test_query}\nexit\n", 
        text=True, 
        capture_output=True, 
        timeout=60
        )
        
        if result.returncode == 0:
            print("Поиск с transformers выполнен успешно")
            # Показываем результат
            output_lines = result.stdout.split('\n')[:15]
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"Ошибка поиска: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Таймаут поиска")
    except Exception as e:
        print(f"Ошибка: {e}")

def main():
    """Основная функция тестирования."""
    print("Начинаем тестирование RAG системы")
    print("=" * 50)
    
    # Проверяем наличие файлов
    if not Path("data/test_data.jsonl").exists():
        print("Файл data/test_data.jsonl не найден")
        return
    
    if not Path("chroma_db").exists():
        print("База данных chroma_db не найдена. Сначала запустите индексацию.")
        return
    
    # Тестируем простой поиск
    test_simple_search()
    
    # Тестируем поиск с transformers
    test_transformers_search()
    
    print("\n" + "=" * 50)
    print("Тестирование завершено")

if __name__ == "__main__":
    main()

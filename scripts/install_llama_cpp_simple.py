"""Простой скрипт для установки llama-cpp-python."""
import subprocess
import sys

def install_llama_cpp():
    """Устанавливает llama-cpp-python."""
    print("Установка llama-cpp-python...")
    
    try:
        # Пробуем установить предварительно скомпилированную версию
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python",
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("llama-cpp-python установлен успешно")
            return True
        else:
            print(f"Ошибка установки: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def test_installation():
    """Тестирует установку."""
    print("Тестируем установку...")
    
    try:
        import llama_cpp
        print(f"llama-cpp-python версия: {llama_cpp.__version__}")
        print("Установка успешна!")
        return True
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        return False

def main():
    """Основная функция."""
    print("Простая установка llama-cpp-python")
    print("=" * 40)
    
    if install_llama_cpp():
        if test_installation():
            print("\nГотово! Теперь можно использовать rag_query.py")
        else:
            print("\nУстановка завершена, но есть проблемы с импортом")
    else:
        print("\nНе удалось установить llama-cpp-python")
        print("Используйте rag_query_transformers.py как альтернативу")

if __name__ == "__main__":
    main()

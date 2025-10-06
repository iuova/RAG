"""Скрипт для автоматической установки llama-cpp-python на Windows."""
import subprocess
import sys
import os
from pathlib import Path

def check_visual_studio():
    """Проверяет наличие Visual Studio Build Tools."""
    print("Проверяем наличие Visual Studio Build Tools...")
    
    # Проверяем наличие cl.exe в PATH
    try:
        result = subprocess.run(['cl'], capture_output=True, text=True)
        if result.returncode == 0 or "Microsoft" in result.stderr:
            print("Visual Studio Build Tools найдены")
            return True
    except FileNotFoundError:
        pass
    
    print("Visual Studio Build Tools не найдены")
    return False

def install_build_tools():
    """Предлагает установить Visual Studio Build Tools."""
    print("\nДля компиляции llama-cpp-python нужны Visual Studio Build Tools.")
    print("Скачайте и установите с: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("Выберите 'C++ build tools' в списке рабочих нагрузок.")
    
    choice = input("\nПродолжить установку llama-cpp-python? (y/n): ").lower()
    return choice == 'y'

def install_precompiled():
    """Устанавливает предварительно скомпилированную версию."""
    print("\nУстанавливаем предварительно скомпилированную версию...")
    
    try:
        # Попробуем установить с предварительно скомпилированными колесами
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

def install_from_source():
    """Устанавливает из исходного кода."""
    print("\nУстанавливаем из исходного кода...")
    
    try:
        # Устанавливаем CMake если нужно
        subprocess.run([sys.executable, "-m", "pip", "install", "cmake"], check=True)
        
        # Компилируем llama-cpp-python
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python", "--no-cache-dir"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ llama-cpp-python скомпилирован и установлен")
            return True
        else:
            print(f"❌ Ошибка компиляции: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def test_installation():
    """Тестирует установку llama-cpp-python."""
    print("\nТестируем установку...")
    
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python версия: {llama_cpp.__version__}")
        
        # Простой тест создания объекта
        from llama_cpp import Llama
        print("✅ Импорт Llama успешен")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

def download_sample_model():
    """Предлагает скачать тестовую модель."""
    print("\nДля тестирования нужна GGUF модель.")
    print("Рекомендуемые модели:")
    print("1. Qwen2.5-7B-Instruct (4.1 GB)")
    print("2. Llama 3.1 8B (4.1 GB)")
    print("3. Mistral 7B (4.1 GB)")
    
    choice = input("\nСкачать тестовую модель? (y/n): ").lower()
    if choice == 'y':
        download_model()

def download_model():
    """Скачивает тестовую модель."""
    print("\nСкачиваем тестовую модель...")
    
    try:
        # Устанавливаем huggingface-hub если нужно
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
        
        from huggingface_hub import hf_hub_download
        
        # Создаем папку models если не существует
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Скачиваем модель
        model_path = hf_hub_download(
            repo_id="TheBloke/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            local_dir=str(models_dir)
        )
        
        print(f"✅ Модель скачана: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка скачивания модели: {e}")
        return False

def main():
    """Основная функция установки."""
    print("Установка llama-cpp-python для RAG системы")
    print("=" * 50)
    
    # Проверяем Python версию
    if sys.version_info < (3, 8):
        print("Требуется Python 3.8 или выше")
        return
    
    print(f"Python версия: {sys.version}")
    
    # Проверяем наличие Visual Studio Build Tools
    has_vs = check_visual_studio()
    
    if not has_vs:
        if not install_build_tools():
            print("Установка отменена")
            return
    
    # Пробуем установить предварительно скомпилированную версию
    if install_precompiled():
        if test_installation():
            print("\n🎉 Установка завершена успешно!")
            download_sample_model()
            return
    
    # Если не получилось, пробуем компиляцию
    if has_vs:
        print("\nПробуем компиляцию из исходного кода...")
        if install_from_source():
            if test_installation():
                print("\n🎉 Установка завершена успешно!")
                download_sample_model()
                return
    
    print("\n❌ Не удалось установить llama-cpp-python")
    print("Рекомендации:")
    print("1. Установите Visual Studio Build Tools")
    print("2. Используйте rag_query_transformers.py как альтернативу")
    print("3. Обратитесь к документации: docs/llama_cpp_installation_guide.md")

if __name__ == "__main__":
    main()

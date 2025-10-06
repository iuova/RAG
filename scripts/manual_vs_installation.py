"""Ручная установка Visual Studio Build Tools - пошаговая инструкция."""
import webbrowser
import os
import subprocess
import sys

def open_download_page():
    """Открывает страницу скачивания Visual Studio Build Tools."""
    print("Открываем страницу скачивания Visual Studio Build Tools...")
    webbrowser.open("https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("Страница открыта в браузере")

def check_installation():
    """Проверяет установку Visual Studio Build Tools."""
    print("Проверяем установку Visual Studio Build Tools...")
    
    try:
        result = subprocess.run(['cl'], capture_output=True, text=True)
        if "Microsoft" in result.stderr or result.returncode == 0:
            print("Visual Studio Build Tools найдены!")
            return True
    except FileNotFoundError:
        pass
    
    print("Visual Studio Build Tools не найдены в PATH")
    return False

def install_llama_cpp():
    """Устанавливает llama-cpp-python."""
    print("Устанавливаем llama-cpp-python...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python", "--no-cache-dir"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("llama-cpp-python установлен успешно!")
            return True
        else:
            print(f"Ошибка установки llama-cpp-python: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def main():
    """Основная функция."""
    print("Ручная установка Visual Studio Build Tools")
    print("=" * 50)
    
    # Проверяем текущую установку
    if check_installation():
        print("Visual Studio Build Tools уже установлены!")
        print("Пробуем установить llama-cpp-python...")
        if install_llama_cpp():
            print("Готово! Теперь можно использовать rag_query.py")
        return
    
    print("Пошаговая инструкция:")
    print("1. Открываем страницу скачивания...")
    open_download_page()
    
    print("\n2. Инструкции по установке:")
    print("   - Скачайте 'Build Tools for Visual Studio 2022'")
    print("   - Запустите установщик")
    print("   - Выберите рабочую нагрузку 'C++ build tools'")
    print("   - Убедитесь, что включены:")
    print("     * MSVC v143 - VS 2022 C++ x64/x86 build tools")
    print("     * Windows 10/11 SDK")
    print("     * CMake tools for Visual Studio")
    print("   - Нажмите 'Установить'")
    print("   - Дождитесь завершения (10-30 минут)")
    
    print("\n3. После установки:")
    print("   - Перезапустите командную строку")
    print("   - Запустите этот скрипт снова")
    print("   - Или выполните: python scripts/install_vs_build_tools.py")
    
    print("\n4. Альтернативы:")
    print("   - Используйте rag_query_transformers.py (работает сразу)")
    print("   - Используйте rag_query_simple.py (быстрый поиск)")
    
    input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()

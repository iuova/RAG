"""Проверка готовности Visual Studio Build Tools."""
import subprocess
import sys

def check_cl_compiler():
    """Проверяет наличие компилятора cl.exe."""
    print("Проверяем компилятор cl.exe...")
    
    try:
        result = subprocess.run(['cl'], capture_output=True, text=True)
        if "Microsoft" in result.stderr or result.returncode == 0:
            print("Компилятор cl.exe найден")
            return True
    except FileNotFoundError:
        pass
    
    print("Компилятор cl.exe не найден")
    return False

def check_cmake():
    """Проверяет наличие CMake."""
    print("Проверяем CMake...")
    
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("CMake найден")
            return True
    except FileNotFoundError:
        pass
    
    print("CMake не найден")
    return False

def check_llama_cpp():
    """Проверяет установку llama-cpp-python."""
    print("Проверяем llama-cpp-python...")
    
    try:
        import llama_cpp
        print(f"llama-cpp-python версия: {llama_cpp.__version__}")
        return True
    except ImportError:
        print("llama-cpp-python не установлен")
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
            print("llama-cpp-python установлен")
            return True
        else:
            print(f"Ошибка установки: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def main():
    """Основная функция проверки."""
    print("Проверка готовности Visual Studio Build Tools")
    print("=" * 50)
    
    # Проверяем компилятор
    cl_ok = check_cl_compiler()
    
    # Проверяем CMake
    cmake_ok = check_cmake()
    
    # Проверяем llama-cpp-python
    llama_ok = check_llama_cpp()
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print(f"Компилятор cl.exe: {'OK' if cl_ok else 'НЕТ'}")
    print(f"CMake: {'OK' if cmake_ok else 'НЕТ'}")
    print(f"llama-cpp-python: {'OK' if llama_ok else 'НЕТ'}")
    
    if cl_ok and cmake_ok:
        print("\nVisual Studio Build Tools готовы!")
        
        if not llama_ok:
            print("Устанавливаем llama-cpp-python...")
            if install_llama_cpp():
                print("Готово! Теперь можно использовать rag_query.py")
            else:
                print("Не удалось установить llama-cpp-python")
        else:
            print("Все готово! Система RAG полностью функциональна")
    else:
        print("\nVisual Studio Build Tools не готовы")
        print("Инструкции по установке:")
        print("1. Скачайте с: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("2. Выберите рабочую нагрузку 'C++ build tools'")
        print("3. Включите MSVC v143, Windows SDK, CMake")
        print("4. Перезапустите командную строку")
        print("5. Запустите этот скрипт снова")
        
        print("\nАльтернативы:")
        print("- Используйте rag_query_transformers.py")
        print("- Используйте rag_query_simple.py")

if __name__ == "__main__":
    main()

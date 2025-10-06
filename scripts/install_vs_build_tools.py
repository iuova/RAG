"""Скрипт для установки Visual Studio Build Tools."""
import subprocess
import sys
import os
import urllib.request
from pathlib import Path

def download_vs_build_tools():
    """Скачивает Visual Studio Build Tools."""
    print("Скачиваем Visual Studio Build Tools...")
    
    # URL для скачивания Visual Studio Build Tools
    url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    installer_path = "vs_buildtools.exe"
    
    try:
        print(f"Скачиваем с {url}")
        urllib.request.urlretrieve(url, installer_path)
        print(f"Установщик сохранен: {installer_path}")
        return installer_path
    except Exception as e:
        print(f"Ошибка скачивания: {e}")
        return None

def install_vs_build_tools(installer_path):
    """Устанавливает Visual Studio Build Tools."""
    print("Запускаем установку Visual Studio Build Tools...")
    
    # Команда для установки с нужными компонентами
    cmd = [
        installer_path,
        "--quiet",
        "--wait",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041",
        "--add", "Microsoft.VisualStudio.Component.CMake.Tools"
    ]
    
    try:
        print("Установка может занять несколько минут...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Visual Studio Build Tools установлены успешно!")
            return True
        else:
            print(f"Ошибка установки: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def check_installation():
    """Проверяет установку Visual Studio Build Tools."""
    print("Проверяем установку...")
    
    try:
        # Проверяем наличие cl.exe
        result = subprocess.run(['cl'], capture_output=True, text=True)
        if "Microsoft" in result.stderr or result.returncode == 0:
            print("Visual Studio Build Tools найдены!")
            return True
    except FileNotFoundError:
        pass
    
    print("Visual Studio Build Tools не найдены в PATH")
    return False

def install_llama_cpp():
    """Устанавливает llama-cpp-python после установки Build Tools."""
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
    """Основная функция установки."""
    print("Установка Visual Studio Build Tools для llama-cpp-python")
    print("=" * 60)
    
    # Проверяем, не установлены ли уже Build Tools
    if check_installation():
        print("Visual Studio Build Tools уже установлены!")
        print("Пробуем установить llama-cpp-python...")
        if install_llama_cpp():
            print("Готово! Теперь можно использовать rag_query.py")
        return
    
    # Скачиваем установщик
    installer_path = download_vs_build_tools()
    if not installer_path:
        print("Не удалось скачать установщик")
        print("Скачайте вручную с: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return
    
    # Устанавливаем Build Tools
    if install_vs_build_tools(installer_path):
        # Удаляем установщик
        try:
            os.remove(installer_path)
            print("Временный файл удален")
        except:
            pass
        
        # Проверяем установку
        if check_installation():
            print("Устанавливаем llama-cpp-python...")
            if install_llama_cpp():
                print("Готово! Теперь можно использовать rag_query.py")
            else:
                print("Build Tools установлены, но llama-cpp-python не установился")
        else:
            print("Build Tools установлены, но не найдены в PATH")
            print("Перезапустите командную строку и попробуйте снова")
    else:
        print("Не удалось установить Visual Studio Build Tools")
        print("Попробуйте установить вручную")

if __name__ == "__main__":
    main()

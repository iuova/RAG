# Установка Visual Studio Build Tools для llama-cpp-python

## Обзор
Visual Studio Build Tools необходимы для компиляции llama-cpp-python на Windows. Данная инструкция поможет установить их автоматически или вручную.

## Автоматическая установка

### Шаг 1: Запуск скрипта
```bash
python scripts/install_vs_build_tools.py
```

Скрипт автоматически:
- Скачает Visual Studio Build Tools
- Установит необходимые компоненты
- Проверит установку
- Установит llama-cpp-python

### Шаг 2: Перезапуск командной строки
После установки перезапустите командную строку для обновления PATH.

## Ручная установка

### Шаг 1: Скачивание установщика
1. Перейдите на https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Скачайте "Build Tools for Visual Studio 2022"
3. Запустите установщик

### Шаг 2: Выбор компонентов
В установщике выберите:
- ✅ **C++ build tools** (основная рабочая нагрузка)
- ✅ **MSVC v143 - VS 2022 C++ x64/x86 build tools**
- ✅ **Windows 10/11 SDK** (последняя версия)
- ✅ **CMake tools for Visual Studio**

### Шаг 3: Установка
1. Нажмите "Установить"
2. Дождитесь завершения (может занять 10-30 минут)
3. Перезапустите командную строку

## Проверка установки

### Тест 1: Проверка компилятора
```bash
cl
```
Должно показать информацию о Microsoft C/C++ компиляторе.

### Тест 2: Проверка в Python
```python
import subprocess
result = subprocess.run(['cl'], capture_output=True, text=True)
print("Компилятор найден:", "Microsoft" in result.stderr)
```

## Установка llama-cpp-python

После установки Build Tools:

```bash
# Установка llama-cpp-python
pip install llama-cpp-python --no-cache-dir

# Или с дополнительными флагами
CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install llama-cpp-python --no-cache-dir
```

## Альтернативные методы

### Метод 1: Через Visual Studio Community
1. Установите Visual Studio Community (бесплатно)
2. Выберите рабочую нагрузку "Desktop development with C++"
3. Установите

### Метод 2: Через Chocolatey
```bash
# Установка Chocolatey (если не установлен)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Установка Build Tools
choco install visualstudio2022buildtools
```

### Метод 3: Через winget
```bash
# Установка через Windows Package Manager
winget install Microsoft.VisualStudio.2022.BuildTools
```

## Устранение проблем

### Проблема 1: "cl не является внутренней или внешней командой"
**Решение:**
1. Перезапустите командную строку
2. Или запустите "Developer Command Prompt for VS"
3. Или добавьте путь к компилятору в PATH

### Проблема 2: Ошибка компиляции llama-cpp-python
**Решение:**
```bash
# Очистите кэш pip
pip cache purge

# Установите с дополнительными флагами
set CMAKE_ARGS=-DLLAMA_CUBLAS=off
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### Проблема 3: Нехватка места на диске
**Решение:**
- Build Tools требуют ~3-4 GB
- Освободите место на диске C:
- Или установите на другой диск

## Проверка готовности системы

### Полный тест
```bash
# Запуск тестирования
python test_llama_cpp.py
```

### Тест RAG системы
```bash
# Тест простого поиска
python rag_query_simple.py

# Тест с transformers
python rag_query_transformers.py

# Тест с llama.cpp (после установки)
python rag_query.py
```

## Размер установки

| Компонент | Размер |
|-----------|--------|
| Visual Studio Build Tools | ~3-4 GB |
| Windows SDK | ~1-2 GB |
| CMake Tools | ~100 MB |
| **Общий размер** | **~4-6 GB** |

## Время установки

- **Скачивание**: 5-15 минут (зависит от скорости интернета)
- **Установка**: 10-30 минут (зависит от производительности)
- **Компиляция llama-cpp-python**: 5-15 минут

## Рекомендации

1. **Установите на SSD** для быстрой работы
2. **Закройте другие программы** во время установки
3. **Используйте стабильное интернет-соединение**
4. **Перезапустите систему** после установки

## Альтернативы

Если установка Build Tools вызывает проблемы:

### 1. Используйте transformers
```bash
python rag_query_transformers.py
```

### 2. Используйте Ollama
```bash
# Установка Ollama
# https://ollama.ai/download

# Запуск модели
ollama run qwen2.5:7b
```

### 3. Используйте предварительно скомпилированные колеса
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## Заключение

После успешной установки Visual Studio Build Tools и llama-cpp-python система RAG будет полностью функциональна со всеми тремя вариантами поиска.

**Статус готовности:**
- ✅ rag_query_simple.py - работает
- ✅ rag_query_transformers.py - работает  
- ✅ rag_query.py - работает (после установки Build Tools)

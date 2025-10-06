# Инструкция по установке llama.cpp для Windows

## Обзор
Данная инструкция поможет установить llama-cpp-python на Windows для работы с RAG системой.

## Требования
- Windows 10/11
- Python 3.8+
- Visual Studio Build Tools или Visual Studio Community
- Git (опционально)

## Метод 1: Установка с предварительно скомпилированными колесами (Рекомендуется)

### Шаг 1: Установка Visual Studio Build Tools
1. Скачайте Visual Studio Build Tools с официального сайта Microsoft
2. Запустите установщик
3. Выберите "C++ build tools" в списке рабочих нагрузок
4. Убедитесь, что включены:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 10/11 SDK
   - CMake tools for Visual Studio
5. Установите компоненты

### Шаг 2: Установка llama-cpp-python
```bash
# Установка с предварительно скомпилированными колесами
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Или для GPU (если есть CUDA)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

## Метод 2: Компиляция из исходного кода

### Шаг 1: Установка зависимостей
```bash
# Установка CMake
pip install cmake

# Установка Visual Studio Build Tools (если не установлены)
# Скачайте с https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Шаг 2: Компиляция llama-cpp-python
```bash
# Для CPU версии
CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install llama-cpp-python --no-cache-dir

# Для GPU версии (если есть CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

## Метод 3: Использование conda (Альтернативный)

### Шаг 1: Установка conda
```bash
# Скачайте и установите Miniconda или Anaconda
# https://docs.conda.io/en/latest/miniconda.html
```

### Шаг 2: Создание окружения
```bash
# Создание нового окружения
conda create -n rag_env python=3.11

# Активация окружения
conda activate rag_env

# Установка llama-cpp-python
conda install -c conda-forge llama-cpp-python
```

## Проверка установки

### Тест 1: Проверка импорта
```python
try:
    import llama_cpp
    print(f"llama-cpp-python версия: {llama_cpp.__version__}")
    print("Установка успешна!")
except ImportError as e:
    print(f"Ошибка импорта: {e}")
```

### Тест 2: Простая загрузка модели
```python
from llama_cpp import Llama

# Тест с простой моделью (если есть)
try:
    llm = Llama(model_path="path/to/model.gguf", n_threads=4, n_ctx=2048)
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
```

## Получение GGUF моделей

### Рекомендуемые источники:
1. **Hugging Face Hub**: https://huggingface.co/models?library=gguf
2. **TheBloke модели**: https://huggingface.co/TheBloke
3. **Ollama**: https://ollama.ai/library

### Популярные модели для RAG:
- **Qwen2.5-7B-Instruct**: `qwen2.5-7b-instruct-q4_k_m.gguf`
- **Llama 3.1 8B**: `llama-3.1-8b-instruct-q4_k_m.gguf`
- **Mistral 7B**: `mistral-7b-instruct-v0.1-q4_k_m.gguf`

### Скачивание модели:
```bash
# Пример скачивания через huggingface-hub
pip install huggingface-hub

# Скачивание модели
python -c "
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id='TheBloke/Qwen2.5-7B-Instruct-GGUF',
    filename='qwen2.5-7b-instruct-q4_k_m.gguf',
    local_dir='./models'
)
print(f'Модель скачана: {model_path}')
"
```

## Настройка RAG системы

### Обновление конфигурации
В файле `config.py` убедитесь, что указан правильный путь к модели:
```python
DEFAULT_MODEL_FILENAME: str = "qwen2.5-7b-instruct-q4_k_m.gguf"
```

### Тестирование RAG с llama.cpp
```bash
# Запуск RAG с llama.cpp
python rag_query.py --model models/qwen2.5-7b-instruct-q4_k_m.gguf
```

## Устранение проблем

### Проблема 1: Ошибка компиляции
```
CMake Error: CMAKE_C_COMPILER not set
```
**Решение**: Установите Visual Studio Build Tools с компонентами C++

### Проблема 2: Ошибка импорта
```
ModuleNotFoundError: No module named 'llama_cpp'
```
**Решение**: Используйте предварительно скомпилированные колеса (Метод 1)

### Проблема 3: Медленная работа
**Решение**: 
- Увеличьте количество потоков: `n_threads=8`
- Используйте квантованную модель (Q4_K_M)
- Уменьшите контекстное окно: `n_ctx=2048`

### Проблема 4: Нехватка памяти
**Решение**:
- Используйте меньшую модель
- Уменьшите контекстное окно
- Используйте квантованную версию

## Альтернативы llama.cpp

Если установка llama.cpp вызывает проблемы, используйте:

### 1. Transformers (уже работает)
```bash
python rag_query_transformers.py
```

### 2. Ollama (простая установка)
```bash
# Установка Ollama
# https://ollama.ai/download

# Запуск модели
ollama run qwen2.5:7b

# Использование через API
```

## Рекомендации по производительности

### Для CPU:
- Используйте Q4_K_M квантование
- Установите `n_threads` равным количеству ядер CPU
- Ограничьте `n_ctx` до 2048-4096 токенов

### Для GPU:
- Установите CUDA версию llama-cpp-python
- Используйте `n_gpu_layers` для загрузки слоев на GPU
- Увеличьте `n_ctx` до 8192+ токенов

## Заключение

После успешной установки llama-cpp-python система RAG будет полностью функциональна со всеми тремя вариантами поиска:
- `rag_query_simple.py` - простой поиск
- `rag_query_transformers.py` - поиск с transformers  
- `rag_query.py` - поиск с llama.cpp

Рекомендуется начать с Метода 1 (предварительно скомпилированные колеса) как наиболее простого.

## Альтернативное решение

Если установка llama-cpp-python вызывает проблемы на Windows, используйте:

### 1. Transformers (рекомендуется)
```bash
python rag_query_transformers.py
```
- Работает сразу после установки зависимостей
- Не требует компиляции
- Поддерживает все современные модели

### 2. Простой поиск
```bash
python rag_query_simple.py
```
- Быстрый поиск без генерации ответов
- Идеально для тестирования и отладки

### 3. Ollama (если нужна локальная генерация)
```bash
# Установка Ollama
# https://ollama.ai/download

# Запуск модели
ollama run qwen2.5:7b
```

## Статус тестирования

**Протестировано на Windows Server 2019:**
- ✅ rag_query_simple.py - работает
- ✅ rag_query_transformers.py - работает  
- ❌ llama-cpp-python - проблемы с компиляцией

**Рекомендация:** Используйте transformers для полной функциональности RAG системы.

# RAG (Retrieval-Augmented Generation) Project

Проект предоставляет офлайн-пайплайн Retrieval-Augmented Generation (RAG) для русскоязычных знаний. Он позволяет индексировать локальные документы в векторную базу Chroma и получать ответы от языковых моделей (через `llama.cpp` или Hugging Face transformers) с опорой на найденные фрагменты.

## Структура репозитория

```
RAG/
├── chroma_db/                # Папка для сохранения векторного индекса (создается автоматически)
├── examples/                 # Примеры входных JSONL-документов
├── llama.cpp/                # Инструкции и скрипты для работы с локальными GGUF-моделями
├── scripts/                  # Служебные сценарии (тестирование модели, подготовка окружения)
├── config.py                 # Конфигурация путей и параметров по умолчанию
├── rag_index.py              # Индексация JSONL-файлов в Chroma
├── rag_query.py              # CLI для генерации ответов с llama.cpp
├── rag_query_transformers.py # CLI для генерации ответов через transformers
├── rag_query_simple.py       # Упрощенный поиск без генерации
├── convert_txt_to_jsonl.py   # Конвертация табличного текста в JSONL
├── simple_convert.py         # Простейший конвертер текстовых файлов
├── xlsx_to_json.py           # Конвертация XLSX в JSONL
├── fix_encoding.py           # Исправление кодировок исходных данных
├── dev-journal.md            # Журнал хода разработки
├── requirements.txt          # Python-зависимости
└── ANALYSIS.md               # Подробный обзор архитектуры проекта
```

## Установка

### Базовые зависимости
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### LLM+RAG без llama.cpp (Рекомендуется)
Для полной функциональности LLM+RAG без Visual Studio Build Tools:

**Готовое решение:**
```bash
# Интеллектуальный RAG с генерацией ответов
python rag_query_final.py --question "ваш вопрос"

# Интерактивный режим
python rag_query_final.py
```

**Альтернативы:**
- `rag_query_hybrid.py` - гибридный подход
- `rag_query_simple.py` - простой поиск
- `rag_query_transformers.py` - с transformers

Подробная инструкция: `docs/llm_rag_without_build_tools.md`

### Установка llama.cpp (опционально)
Для работы с llama.cpp (требует Visual Studio Build Tools):

**Автоматическая установка:**
```bash
python scripts/install_llama_cpp.py
```

Подробная инструкция: `docs/llama_cpp_installation_guide.md`

## Быстрый старт

2. **Подготовка данных**
   - Сформируйте JSONL-файл с документами вида:
     ```json
     {"id": "doc-1", "text": "Текст документа", "metadata": {"source": "пример"}}
     ```
   - Пример минимального набора лежит в `examples/example_documents.jsonl`.

3. **Построение индекса**
   ```bash
   python rag_index.py --source examples/example_documents.jsonl --collection demo
   ```
   Скрипт создаст/обновит коллекцию Chroma в `chroma_db/`.

4. **Запрос с генерацией ответа**
   - **Для llama.cpp**: Установите llama-cpp-python и подготовьте GGUF-модель (см. `docs/llama_cpp_installation_guide.md`)
   - **Для transformers**: Используйте `rag_query_transformers.py` (работает сразу)
   - **Простой поиск**: Используйте `rag_query_simple.py` (без генерации)
   - Запустите CLI:
     ```bash
     python rag_query.py --collection demo --question "Что содержит пример?"
     ```
   - Для CPU-билда без llama.cpp используйте `rag_query_transformers.py`:
     ```bash
     python rag_query_transformers.py --collection demo --question "Что содержит пример?"
     ```

5. **Поиск без генерации**
   ```bash
   python rag_query_simple.py --collection demo --question "Что содержит пример?"
   ```
   Скрипт выведет k наиболее релевантных документов.

## Часто задаваемые вопросы

### Поддерживает ли проект RAG?
Да. Все основные скрипты направлены на реализацию Retrieval-Augmented Generation: индексатор (`rag_index.py`) сохраняет векторы документов в Chroma, а запросные CLI (`rag_query.py`, `rag_query_transformers.py`, `rag_query_simple.py`) извлекают релевантные контексты и при необходимости генерируют ответ языковой моделью.

## Дополнительные ресурсы
- Подробный обзор архитектуры и рекомендаций: `ANALYSIS.md`.
- История и заметки по эксплуатации: `dev-journal.md`.


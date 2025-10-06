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

## Быстрый старт

1. **Установка зависимостей**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

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
   - Подготовьте GGUF-модель и распакуйте её в каталог `models/` (см. инструкции в `llama.cpp/README.md`).
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


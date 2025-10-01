# Локальный RAG без GPU

Этот репозиторий содержит полностью автономный пайплайн Retrieval-Augmented Generation (RAG)
для запуска на **Windows Server 2019** с CPU (без GPU). Система рассчитана на железо уровня
Intel Xeon с большим количеством оперативной памяти (например, 1 ТБ) и использует модель
формата GGUF через [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

## Возможности

- **Индексация документов** (JSONL, TXT/Markdown) в локальную базу ChromaDB с сохранением метаданных
- **Интерактивный консольный чат** с ответами на основе извлечённого контекста и ссылками на источники
- Работа **целиком офлайн** после единовременной загрузки модели
- Оптимизировано для **CPU**: embeddings HuggingFace + генерация через llama.cpp
- Вспомогательные скрипты для подготовки данных и автоматизированной настройки среды

## Структура

```
RAG/
├── config.py               # Общая конфигурация путей и параметров
├── rag_index.py            # Индексация документов в ChromaDB
├── rag_query.py            # Консольный RAG-чат поверх llama-cpp-python
├── xlsx_to_json.py         # Конвертация Excel → JSONL
├── examples/               # Примеры данных и шаблонов
│   └── example_documents.jsonl
├── llama.cpp/              # Бинарники llama.cpp (по желанию)
├── scripts/
│   ├── setup_windows.ps1   # Полная автоматическая настройка под Windows
│   └── test_model.py       # Смоук-тест модели GGUF
├── logs/                   # Логи работы (создаются автоматически)
├── chroma_db/              # Каталог векторной БД (создаётся при индексации)
├── data/                   # Пользовательские данные (в .gitignore)
├── models/                 # Загруженные GGUF-модели (в .gitignore)
└── requirements.txt        # Зависимости Python
```

## Подготовка окружения (Windows Server 2019, CPU)

1. Установите **Python 3.10+** (64-bit) и добавьте его в `PATH`.
2. Склонируйте репозиторий и перейдите в папку проекта:
   ```powershell
   git clone https://github.com/iuova/RAG.git
   cd RAG
   ```
3. Убедитесь, что в системе установлен [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) (необходим для бинарей llama.cpp).
4. Запустите автоматический скрипт настройки (скачается модель ~5.5 ГБ):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
   ```
   Скрипт создаст виртуальное окружение `venv`, установит зависимости, скачает
   модель **TheBloke/Qwen2.5-7B-Instruct-GGUF (q4_k_m)** и сохранит её в `models/`.
5. Активируйте окружение:
   ```powershell
   .\venv\Scripts\activate
   ```

> **Примечание:** если доступ к Hugging Face ограничен, скачайте GGUF-файл вручную и
> поместите его в `models/`, затем укажите путь через `--model` при запуске `rag_query.py`.

### Ручная установка зависимостей

Если автоматический скрипт недоступен, выполните шаги вручную:

```powershell
python -m venv venv
./venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

После установки зависимостей скачайте GGUF-модель через `huggingface_hub`:

```powershell
python -m huggingface_hub download `
    --repo-id TheBloke/Qwen2.5-7B-Instruct-GGUF `
    --include "*q4_k_m.gguf" `
    --local-dir models `
    --local-dir-use-symlinks False
```

## Подготовка данных

1. Скопируйте собственные документы в формат **JSONL** или **TXT/MD** в каталог `data/`.
2. Для Excel-файлов используйте конвертер:
   ```powershell
   python xlsx_to_json.py path\to\source.xlsx --output data\my_docs.jsonl
   ```
   Колонки должны называться `Работа`, `ПунктРемонтнойВедомости`, `Описание`.
3. Для быстрого старта можно использовать пример:
   ```powershell
   copy examples\example_documents.jsonl data\data.jsonl
   ```

## Индексация

```powershell
python rag_index.py data\data.jsonl
```

Параметры:
- `inputs` — список файлов JSONL/TXT. Если не указывать, используется `data/data.jsonl`.
- `--collection` — имя коллекции Chroma (по умолчанию `docs`).
- `--batch-size` — размер батча при записи чанков.
- `--reset` — удалить существующую коллекцию перед индексацией.

Результат сохранится в каталоге `chroma_db/`.

## Запросы к модели

```powershell
python rag_query.py
```

Параметры:
- `--model` — путь до GGUF-файла (по умолчанию `models/qwen2.5-7b-instruct-q4_k_m.gguf`).
- `--threads` — количество CPU-потоков для инференса (для Xeon Gold 6330H можно ставить 48-64).
- `--ctx` — размер контекстного окна (`4096` по умолчанию).
- `--top-k` — число документов, извлекаемых из Chroma.

Команда откроет интерактивную консоль. Введите вопрос или `exit` для выхода. В ответе указываются номера чанков и источники исходных файлов.

## Проверка модели без RAG

```powershell
python scripts/test_model.py
```

Скрипт выдаст короткий ответ, удостоверив, что модель и llama.cpp работают корректно.

## Настройка параметров

Основные значения (пути, размеры чанков, имена моделей) вынесены в `config.py` и могут
переопределяться переменными окружения `RAG_*`. Например, чтобы использовать другую
эмбеддинговую модель:

```powershell
$env:RAG_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
python rag_index.py my_docs.jsonl
```

## Рекомендации по производительности

- Для сервера с 4 физическими CPU (Xeon Gold 6330H) используйте `--threads 96` при наличии
  гиперпоточности.
- Формат `q4_k_m.gguf` занимает ~5.5 ГБ и комфортно работает на CPU. При наличии свободной
  RAM можно попробовать `q5_k_m` или `q6_k`.
- При больших объёмах данных увеличивайте `RAG_BATCH_SIZE` (например, до 256) и
  `RAG_CHUNK_SIZE` (например, 1200-1500 символов).

## Устранение неполадок

- **ImportError llama_cpp** — убедитесь, что установка выполнялась через CPU-репозиторий
  колёс: `pip install llama-cpp-python --extra-index-url ...` (см. скрипт).
- **RuntimeError: model not found** — проверьте путь до GGUF-файла или передайте его явно
  через `--model`.
- **Пустые ответы** — удостоверьтесь, что индекс создан и коллекция не пуста (`chroma_db/`).

## Лицензии

Модели GGUF распространяются согласно лицензиям их авторов. См. страницу релиза на
Hugging Face перед использованием в продакшене.

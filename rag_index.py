import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------- Конфигурация -------------
DATA_JSONL = Path(r"C:\Users\O.Iunina\Desktop\Projects\RAG\data\data.jsonl")
CHROMA_DB_DIR = Path(r"C:\Users\O.Iunina\Desktop\Projects\RAG\chroma_db")
COLLECTION = "docs"
LOG_FILE = Path(r"C:\Users\O.Iunina\Desktop\Projects\RAG\rag_index.log")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 100

# ------------- Логирование -------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("=== Запуск индексации ===")

# ------------- Проверка данных -------------
if not DATA_JSONL.exists():
    logging.error(f"Файл {DATA_JSONL} не найден")
    raise FileNotFoundError(f"Файл {DATA_JSONL} не найден")

# ------------- Загрузка модели -------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ------------- Подключение Chroma -------------
vectorstore = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    collection_name=COLLECTION,
    embedding_function=embedding_model
)

# ------------- Разбивка текста на чанки -------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True
)

logging.info(f"Загрузка данных из {DATA_JSONL}")
print(f"Загрузка данных из {DATA_JSONL}...")

documents = []
with open(DATA_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or ""
            if text.strip():
                documents.append(text)
        except json.JSONDecodeError:
            logging.warning("Ошибка JSON в строке, пропускаем")

logging.info(f"Всего документов: {len(documents)}")
print(f"Всего документов: {len(documents)}")

# ------------- Чанкинг -------------
chunks = []
for doc in tqdm(documents, desc="Разбивка на чанки"):
    chunks.extend(text_splitter.split_text(doc))

logging.info(f"Всего чанков: {len(chunks)}")
print(f"Всего чанков: {len(chunks)}")

# ------------- Индексация (пакетами) -------------
print("Добавляем данные в Chroma...")
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Индексация"):
    batch = chunks[i:i+BATCH_SIZE]
    vectorstore.add_texts(batch)

# ------------- Сохранение -------------
vectorstore.persist()
logging.info("✅ Индексация завершена")
print(f"✅ Индексация завершена. База сохранена в: {CHROMA_DB_DIR}")

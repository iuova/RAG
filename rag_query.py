import logging
from pathlib import Path

from langchain_chroma import Chroma  # Новый актуальный импорт
from langchain_ollama import OllamaLLM  # Новый актуальный импорт для Ollama
from langchain_huggingface import HuggingFaceEmbeddings  # Новый актуальный импорт
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Пути
CHROMA_DB_DIR = Path(r"C:\Users\O.Iunina\Desktop\Projects\RAG\chroma_db")
LOG_FILE = Path(r"C:\Users\O.Iunina\Desktop\Projects\RAG\rag.log")
COLLECTION = "docs_collection"

# Логирование
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

def get_llm():
    """LLM через Ollama (актуальный импорт)"""
    return OllamaLLM(model="llama3:8b")  # Запуск через Ollama

def get_embeddings():
    """HuggingFace embeddings (актуальный импорт)"""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

def main():
    embeddings = get_embeddings()
    llm = get_llm()

    # Подключение к Chroma
    db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name=COLLECTION,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Шаблон для RAG
    template = """Используя предоставленные контексты, ответь на вопрос пользователя.
Контекст:
{context}
Вопрос:
{question}
Ответ на русском:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    while True:
        query = input("Введите вопрос (или 'exit' для выхода): ")
        if query.lower() == "exit":
            break

        result = qa(query)
        answer = result["result"]
        sources = [doc.metadata.get("source", "Неизвестно") for doc in result["source_documents"]]

        print("\n=== Ответ ===")
        print(answer)
        print("\n=== Источники ===")
        for src in sources:
            print(f"- {src}")

        # Логирование
        logging.info(f"Вопрос: {query}")
        logging.info(f"Ответ: {answer}")
        logging.info(f"Источники: {sources}\n")

if __name__ == "__main__":
    main()

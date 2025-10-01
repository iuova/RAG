"""Run interactive RAG queries against a local llama.cpp model."""
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import List

from llama_cpp import Llama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DB_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_NUM_THREADS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    LOG_DIR,
    MODELS_DIR,
    ensure_directories,
)


def format_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"Источник {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts))
    return textwrap.dedent(
        f"""
        Ты — экспертный технический ассистент. Отвечай на русском языке.
        Используй только факты из приведённого контекста. Если информации недостаточно,
        сообщи об этом честно.

        Контекст:
        {context_block}

        Вопрос: {question}
        Развёрнутый ответ:
        """
    ).strip()


def run_query(llm: Llama, retriever: Chroma, question: str, top_k: int) -> None:
    docs = retriever.similarity_search(question, k=top_k)
    if not docs:
        print("Документов по запросу не найдено. Убедитесь, что индекс создан.")
        return

    contexts = [doc.page_content for doc in docs]
    prompt = format_prompt(question, contexts)

    output = llm.create_completion(
        prompt=prompt,
        max_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        stop=["</s>", "\n\nВопрос"],
    )
    answer = output["choices"][0]["text"].strip()

    print("\n=== Ответ ===")
    print(answer)
    print("\n=== Источники ===")
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("source_path") or "неизвестно"
        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            print(f"- {source} (chunk {chunk_index})")
        else:
            print(f"- {source}")

    logging.info("Вопрос: %s", question)
    logging.info("Ответ: %s", answer)
    logging.info("Источники: %s", [doc.metadata for doc in docs])


def build_llm(model_path: Path, n_threads: int, n_ctx: int) -> Llama:
    return Llama(
        model_path=str(model_path),
        n_threads=n_threads,
        n_ctx=n_ctx,
        verbose=False,
    )


def build_retriever(collection_name: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI for local RAG queries.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / DEFAULT_MODEL_FILENAME,
        help=(
            "Path to the GGUF model file. Defaults to models/"
            f"{DEFAULT_MODEL_FILENAME}."
        ),
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name to query.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=(
            "Number of CPU threads for inference. Defaults to RAG_NUM_THREADS or "
            f"{DEFAULT_NUM_THREADS}."
        ),
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context window size for the model (n_ctx).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of documents to retrieve per query. Defaults to RAG_TOP_K or "
            f"{DEFAULT_TOP_K}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    log_path = LOG_DIR / "rag_query.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}. Please download a GGUF model into the models directory."
        )

    llm = build_llm(args.model, args.threads, args.ctx)
    retriever = build_retriever(args.collection)

    print("Интерактивный режим. Введите 'exit' для выхода.")
    while True:
        question = input("\nВаш вопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            break
        if not question:
            continue
        run_query(llm, retriever, question, args.top_k)


if __name__ == "__main__":
    main()

"""Create or update a ChromaDB index for the local RAG pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm
from chromadb import PersistentClient
from embedding_utils import HuggingFaceEncoder

from config import (
    CHROMA_DB_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JSONL,
    LOG_DIR,
    ensure_directories,
)


@dataclass(frozen=True)
class DocumentRecord:
    """A single source document with optional metadata."""

    text: str
    metadata: dict


def load_jsonl_documents(path: Path) -> List[DocumentRecord]:
    """Load text entries from a JSONL file."""

    documents: List[DocumentRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("JSON decode error in %s:%s: %s", path, line_number, exc)
                continue

            text = obj.get("text") or obj.get("content") or obj.get("body")
            if not isinstance(text, str) or not text.strip():
                continue

            metadata = {
                "source": path.name,
                "source_path": str(path),
                "line": line_number,
            }
            if "id" in obj:
                metadata["document_id"] = str(obj["id"])
            if "title" in obj and obj["title"]:
                metadata["title"] = str(obj["title"])

            documents.append(DocumentRecord(text=text.strip(), metadata=metadata))
    return documents


def iter_documents(paths: Iterable[Path]) -> List[DocumentRecord]:
    """Load documents from a list of files."""

    docs: List[DocumentRecord] = []
    for path in paths:
        if not path.exists():
            logging.warning("File %s not found, skipping", path)
            continue
        if path.suffix.lower() == ".jsonl":
            docs.extend(load_jsonl_documents(path))
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            docs.append(
                DocumentRecord(
                    text=text,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                    },
                )
            )
        else:
            logging.warning("Unsupported file extension for %s, skipping", path)
    return docs


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks without external dependencies.

    The logic mimics the behaviour of the old LangChain splitter by
    respecting the configured chunk size and overlap. The implementation
    keeps whitespace boundaries when possible so that the resulting chunks
    remain readable while guaranteeing forward progress.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[str] = []
    text_length = len(normalized)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = normalized[start:end]

        if end < text_length:
            # Try to keep whole words by moving the cut to the last space
            # within the chunk, but only if this still leaves at least
            # 60 % of the original chunk to avoid tiny fragments.
            relative_limit = int(chunk_size * 0.6)
            split_at = chunk.rfind(" ")
            if relative_limit > 0 and split_at >= relative_limit:
                end = start + split_at
                chunk = normalized[start:end]

        chunks.append(chunk.strip())

        if end >= text_length:
            break
        start = max(end - chunk_overlap, 0)

    return [c for c in chunks if c]


def chunk_documents(documents: Sequence[DocumentRecord]) -> tuple[List[str], List[dict]]:
    """Split documents into chunks while preserving metadata."""

    chunk_texts: List[str] = []
    chunk_metadata: List[dict] = []

    for doc_index, document in enumerate(
        tqdm(documents, desc="Разбивка на чанки", unit="док")
    ):
        splits = split_text(
            document.text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
        for chunk_index, chunk in enumerate(splits):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_id": f"{metadata.get('document_id', metadata.get('source', doc_index))}:{chunk_index}",
                }
            )
            chunk_texts.append(chunk)
            chunk_metadata.append(metadata)

    return chunk_texts, chunk_metadata


def build_vector_store(
    documents: Sequence[DocumentRecord], collection_name: str, batch_size: int
) -> None:
    """Persist documents inside a Chroma collection."""

    if not documents:
        raise ValueError("No documents were loaded for indexing.")

    chunk_texts, chunk_metadata = chunk_documents(documents)
    logging.info("Prepared %s chunks", len(chunk_texts))

    logging.info(
        "Loading embedding model %s on CPU", DEFAULT_EMBEDDING_MODEL
    )
    encoder = HuggingFaceEncoder(DEFAULT_EMBEDDING_MODEL, device="cpu")

    logging.info("Encoding %s chunks", len(chunk_texts))
    embeddings_array = encoder.encode(chunk_texts, batch_size=batch_size)

    client = PersistentClient(path=str(CHROMA_DB_DIR))
    logging.info("Preparing Chroma collection '%s'", collection_name)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name)

    logging.info(
        "Adding %s chunks to Chroma in batches of %s", len(chunk_texts), batch_size
    )
    for start in tqdm(
        range(0, len(chunk_texts), batch_size),
        desc="Индексация",
        unit="чанк",
    ):
        end = start + batch_size
        batch_texts = chunk_texts[start:end]
        batch_metadata = chunk_metadata[start:end]
        batch_ids = [
            str(meta.get("chunk_id", idx))
            for idx, meta in enumerate(batch_metadata, start=start)
        ]
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata,
            embeddings=embeddings_array[start:end].tolist(),
        )

    logging.info("Index persisted to %s", CHROMA_DB_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local Chroma index for RAG.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_JSONL],
        help="Files to ingest (JSONL or TXT). Defaults to data/data.jsonl.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of chunks per write batch. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Удалить существующую коллекцию перед индексацией.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    log_path = LOG_DIR / "rag_index.log"
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
    logging.info("Starting indexing run")
    logging.info("Input files: %s", ", ".join(str(p) for p in args.inputs))

    if args.reset and CHROMA_DB_DIR.exists():
        logging.info("Resetting Chroma directory %s", CHROMA_DB_DIR)
        shutil.rmtree(CHROMA_DB_DIR)
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    documents = iter_documents(args.inputs)
    logging.info("Loaded %s source documents", len(documents))

    build_vector_store(documents, args.collection, args.batch_size)
    print(f"Индексация завершена. База сохранена в: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()

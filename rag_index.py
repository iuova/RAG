"""Create or update a ChromaDB index for the local RAG pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, List

from tqdm import tqdm
from chromadb import PersistentClient
from embedding_utils import batch_iterable, get_encoder

from config import (
    CHROMA_DB_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JSONL,
    LOG_DIR,
    ensure_directories,
)

try:
    import ijson
except ImportError:  # pragma: no cover - optional at runtime but required in prod
    ijson = None


@dataclass(frozen=True)
class DocumentRecord:
    """A single source document with optional metadata."""

    text: str
    metadata: dict


def _derive_document_id(
    payload: dict, fallback_prefix: str, index: int, text: str
) -> str:
    """Return a stable identifier for a document payload."""

    preferred_keys = ("document_id", "id", "doc_id", "uuid", "code")
    for key in preferred_keys:
        value = payload.get(key)
        if value:
            return str(value)

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324
    return f"{fallback_prefix}-{index}-{digest[:12]}"


def _coerce_entries(obj: object) -> List[dict]:
    """Extract a list of document-like dictionaries from arbitrary JSON."""

    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]

    if isinstance(obj, dict):
        candidate_keys = ("data", "items", "records", "documents", "rows")
        for key in candidate_keys:
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [obj]

    return []


def _extract_text_field(payload: dict) -> str | None:
    """Find the most likely text field inside a payload."""

    text_keys = (
        "text",
        "content",
        "body",
        "answer",
        "description",
        "document",
        "data",
    )
    for key in text_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_jsonl_documents(path: Path) -> Iterator[DocumentRecord]:
    """Stream text entries from a JSONL file without buffering everything."""

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

            document_id = _derive_document_id(
                obj, path.stem, line_number, text.strip()
            )
            metadata = {
                "source": path.name,
                "source_path": str(path),
                "line": line_number,
                "document_id": document_id,
            }
            if "title" in obj and obj["title"]:
                metadata["title"] = str(obj["title"])

            yield DocumentRecord(text=text.strip(), metadata=metadata)


def _first_non_whitespace_character(path: Path) -> str | None:
    """Return the first non-whitespace character of a file or ``None``."""

    with path.open("r", encoding="utf-8") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                return None
            for char in chunk:
                if not char.isspace():
                    return char


def _iter_json_payload(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield dictionary payloads from JSON files using streaming parsers."""

    if ijson is None:
        logging.warning(
            "ijson не установлен. Файл %s будет прочитан целиком в память.", path
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logging.error("Не удалось прочитать JSON из %s: %s", path, exc)
            return
        for index, item in enumerate(_coerce_entries(payload), start=1):
            if isinstance(item, dict):
                yield index, item
        return

    first_char = _first_non_whitespace_character(path)
    if first_char == "[":
        candidate_streams = ["item"]
    else:
        candidate_streams = [
            "data.item",
            "items.item",
            "records.item",
            "documents.item",
            "rows.item",
            "item",
        ]

    for prefix in candidate_streams:
        yielded_any = False
        try:
            with path.open("rb") as file:
                for index, item in enumerate(ijson.items(file, prefix), start=1):
                    if not isinstance(item, dict):
                        continue
                    yielded_any = True
                    yield index, item
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.debug(
                "Не удалось обработать %s с префиксом %s: %s", path, prefix, exc
            )
            continue
        if yielded_any:
            return

    logging.debug(
        "Не найден подходящий массив в %s, используется полное чтение файла", path
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.error("Не удалось прочитать JSON из %s: %s", path, exc)
        return

    if isinstance(payload, dict):
        yield 1, payload
    elif isinstance(payload, list):
        for index, item in enumerate(payload, start=1):
            if isinstance(item, dict):
                yield index, item


def load_json_documents(path: Path) -> Iterator[DocumentRecord]:
    """Stream text entries from a generic JSON file using a streaming parser."""

    yielded = False
    for index, item in _iter_json_payload(path):
        text = _extract_text_field(item)
        if not text:
            logging.debug(
                "Запись %s из %s пропущена: отсутствует текстовое поле", index, path
            )
            continue

        document_id = _derive_document_id(item, path.stem, index, text)
        metadata = {
            "source": path.name,
            "source_path": str(path),
            "document_id": document_id,
        }
        if "title" in item and item["title"]:
            metadata["title"] = str(item["title"])

        yielded = True
        yield DocumentRecord(text=text, metadata=metadata)

    if not yielded:
        logging.warning(
            "JSON файл %s не содержит подходящих записей для индексации", path
        )


def iter_documents(paths: Iterable[Path]) -> Iterator[DocumentRecord]:
    """Yield documents from a list of files without accumulating everything."""

    for path in paths:
        if not path.exists():
            logging.warning("File %s not found, skipping", path)
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from load_jsonl_documents(path)
        elif suffix == ".json":
            yield from load_json_documents(path)
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            yield DocumentRecord(
                text=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                },
            )
        else:
            logging.warning("Unsupported file extension for %s, skipping", path)


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


def iter_document_chunks(
    documents: Iterable[DocumentRecord],
) -> Iterator[tuple[str, dict]]:
    """Yield chunks of all documents one by one to avoid large buffers."""

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
            chunk_id_base = metadata.get(
                "document_id", metadata.get("source", str(doc_index))
            )
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_id": f"{chunk_id_base}:{chunk_index}",
                }
            )
            yield chunk, metadata


def build_vector_store(
    documents: Iterable[DocumentRecord],
    collection_name: str,
    batch_size: int,
    device: str,
    reset: bool,
) -> None:
    """Persist documents inside a Chroma collection with streaming batches."""

    logging.info(
        "Loading embedding model %s on %s", DEFAULT_EMBEDDING_MODEL, device
    )
    encoder = get_encoder(DEFAULT_EMBEDDING_MODEL, device=device)

    client = PersistentClient(path=str(CHROMA_DB_DIR))
    logging.info("Preparing Chroma collection '%s'", collection_name)
    if reset:
        try:
            client.delete_collection(collection_name)
            logging.info("Existing collection '%s' removed", collection_name)
        except Exception:
            logging.info("Collection '%s' did not exist before reset", collection_name)
    collection = client.get_or_create_collection(name=collection_name)

    documents_iter = iter(documents)
    documents_processed = 0

    def counted_documents() -> Iterator[DocumentRecord]:
        nonlocal documents_processed
        for document in documents_iter:
            documents_processed += 1
            yield document

    chunk_stream = iter_document_chunks(counted_documents())
    total_chunks = 0
    skipped_duplicates = 0

    for batch in tqdm(
        batch_iterable(chunk_stream, batch_size),
        desc="Индексация",
        unit="чанк",
    ):
        batch_texts = [item[0] for item in batch]
        batch_metadata = [item[1] for item in batch]
        batch_ids = [
            str(meta.get("chunk_id", f"chunk-{total_chunks + idx}"))
            for idx, meta in enumerate(batch_metadata)
        ]
        embeddings = encoder.encode(batch_texts, batch_size=len(batch_texts))
        payload = {
            "ids": batch_ids,
            "documents": batch_texts,
            "metadatas": batch_metadata,
            "embeddings": embeddings.tolist(),
        }

        if hasattr(collection, "upsert"):
            collection.upsert(**payload)
            total_chunks += len(batch_texts)
        else:
            try:
                existing = collection.get(ids=batch_ids)
                existing_ids = set(existing.get("ids", [])) if existing else set()
            except Exception:
                logging.exception("Не удалось проверить наличие дубликатов в коллекции")
                existing_ids = set()

            filtered = [
                (
                    payload["ids"][idx],
                    payload["documents"][idx],
                    payload["metadatas"][idx],
                    payload["embeddings"][idx],
                )
                for idx in range(len(batch_ids))
                if payload["ids"][idx] not in existing_ids
            ]

            if not filtered:
                skipped_duplicates += len(batch_ids)
                continue

            if existing_ids:
                skipped_duplicates += len(existing_ids)

            collection.add(
                ids=[item[0] for item in filtered],
                documents=[item[1] for item in filtered],
                metadatas=[item[2] for item in filtered],
                embeddings=[item[3] for item in filtered],
            )
            total_chunks += len(filtered)

    if documents_processed == 0:
        raise ValueError("No documents were loaded for indexing.")

    if total_chunks == 0:
        logging.warning("No chunks were generated from the provided documents.")
    else:
        logging.info("Persisted %s chunks to %s", total_chunks, CHROMA_DB_DIR)
    logging.info("Обработано %s документов", documents_processed)
    if skipped_duplicates:
        logging.info("Пропущено %s чанков из-за дубликатов", skipped_duplicates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local Chroma index for RAG.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_JSONL],
        help="Files to ingest (JSON/JSONL/TXT). Defaults to data/data_for_RAG.json.",
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
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Устройство для расчёта эмбеддингов (cpu, cuda, auto).",
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

    raw_documents = iter_documents(args.inputs)
    raw_iterator = iter(raw_documents)
    try:
        first_document = next(raw_iterator)
    except StopIteration:
        logging.error("Нет документов для индексации по указанным путям")
        print("Не найдено документов для индексации. Проверьте входные файлы.")
        return

    documents = chain([first_document], raw_iterator)
    logging.info(
        "Начата потоковая загрузка документов. Первый источник: %s",
        first_document.metadata.get("source", first_document.metadata.get("source_path", "<unknown>")),
    )

    try:
        build_vector_store(
            documents,
            args.collection,
            args.batch_size,
            args.device,
            args.reset,
        )
    except ValueError as exc:
        logging.error("Indexing aborted: %s", exc)
        print(f"Индексация остановлена: {exc}")
        return
    except Exception as exc:
        logging.exception("Unexpected error during indexing")
        print(f"Произошла ошибка при индексации: {exc}")
        return

    print(f"Индексация завершена. База сохранена в: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()

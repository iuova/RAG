# Repository Analysis: RAG Project

## Overview
This repository implements a local Retrieval-Augmented Generation (RAG) pipeline for Russian-language knowledge bases. The project centres around building and querying a Chroma vector database, with multiple front-ends for retrieval and answer generation. Configuration, ingestion, and query components are designed to run offline with CPU-only hardware.

## Core Components
- **Configuration (`config.py`)** – Defines filesystem layout, default hyperparameters, and helper utilities to ensure required directories exist.【F:config.py†L1-L47】
- **Indexing (`rag_index.py`)** – Loads JSONL/CSV-like sources, chunks documents with `RecursiveCharacterTextSplitter`, embeds using HuggingFace models, and persists vectors to Chroma. Supports batch ingestion and collection resets.【F:rag_index.py†L1-L218】
- **Query Interfaces**:
  - `rag_query.py`: Interactive CLI backed by `llama.cpp` models for answer generation, retrieving top-k documents from Chroma.【F:rag_query.py†L1-L168】
  - `rag_query_simple.py`: Lightweight retriever-only CLI for environments without LLM generation.【F:rag_query_simple.py†L1-L114】
  - `rag_query_transformers.py`: Alternative pipeline using Hugging Face `transformers` for generation on CPU.【F:rag_query_transformers.py†L1-L132】
- **Data Conversion Utilities** – Scripts to convert tabular text and Excel sources into the JSONL format expected by the indexer (`convert_txt_to_jsonl.py`, `simple_convert.py`, `xlsx_to_json.py`).【F:convert_txt_to_jsonl.py†L1-L94】【F:simple_convert.py†L1-L54】【F:xlsx_to_json.py†L1-L66】
- **Maintenance Scripts** – Encoding fix helper (`fix_encoding.py`) and automation utilities in `scripts/` (Windows setup, model tests).【F:fix_encoding.py†L1-L11】【F:scripts/test_model.py†L1-L200】
- **Project Journal** – `dev-journal.md` documents chronological project milestones, data ingestion steps, and operational notes.【F:dev-journal.md†L1-L168】

## Data & Resources
- **Examples** – `examples/example_documents.jsonl` provides minimal sample records for testing ingestion.【F:examples/example_documents.jsonl†L1-L2】
- **Persistent Artifacts** – Local directories for embeddings (`chroma_db/`), models (`models/`), and logs are referenced but Git-ignored to keep the repo lean.

## Dependencies & Tooling
- `requirements.txt` lists a broad stack covering embeddings, vector stores, FastAPI, and development utilities; actual scripts currently rely primarily on LangChain, Hugging Face, ChromaDB, and tqdm.【F:requirements.txt†L1-L33】
- The project targets CPU execution (`device='cpu'` in embeddings, `llama_cpp` configuration) and expects GGUF quantized models for inference.【F:rag_query.py†L70-L128】

## Observations
1. **Documentation Gap** – README describes an idealized structure (`src/`, `tests/`, etc.) that differs from the current file layout. Consolidating docs to match reality would reduce onboarding friction.【F:README.md†L6-L33】
2. **Testing Coverage** – No automated tests exist despite references in `requirements.txt`; consider adding regression tests for conversion scripts and indexing logic.
3. **Large Dependencies** – `requirements.txt` includes packages (e.g., `pinecone-client`, FastAPI) not used in committed scripts, suggesting an opportunity to slim down or split requirements.
4. **Data Expectations** – Several scripts assume external data files (e.g., `data/Data_5000 .txt`) which are Git-ignored; providing download instructions or synthetic datasets would help reproducibility.

## Recommendations
- Align documentation with the actual repository layout and add quickstart instructions for generating a small index end-to-end.
- Introduce automated validation (unit tests or CLI smoke tests) for the ingestion and query flows.
- Modularize dependency management (core vs. optional) to streamline setup on constrained environments.
- Provide example datasets or scripted download hooks to bootstrap the system without private data.


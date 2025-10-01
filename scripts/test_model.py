"""Smoke-test the local llama.cpp model after installation."""
from __future__ import annotations

import argparse
from pathlib import Path

from llama_cpp import Llama

from config import DEFAULT_MODEL_FILENAME, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly verify llama.cpp inference.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / DEFAULT_MODEL_FILENAME,
        help="Path to the GGUF file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="CPU threads to use during inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    llm = Llama(model_path=str(args.model), n_threads=args.threads, n_ctx=2048, verbose=False)
    prompt = "Коротко опиши возможности Windows Server 2019."
    result = llm.create_completion(prompt=prompt, max_tokens=200, temperature=0.1)
    print(result["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()

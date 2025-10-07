"""Utility helpers for loading and using embedding models."""
from __future__ import annotations

from functools import lru_cache
import logging
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _normalise_device(device: str | None = None) -> str:
    """Resolve a device string and provide helpful fallbacks."""

    requested = (device or "auto").lower()
    if requested in {"auto", "default"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested


class HuggingFaceEncoder:
    """Thin wrapper around Hugging Face transformer models for embeddings."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512) -> None:
        resolved_device = _normalise_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(resolved_device)
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into L2-normalised embeddings."""

        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                token_counts = attention_mask.sum(dim=1).clamp(min=1e-6)
                pooled = sum_hidden / token_counts
                normalised = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(normalised.cpu().numpy())

        return np.vstack(embeddings)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string and return a 1D numpy array."""

        return self.encode([text], batch_size=1)[0]


@lru_cache(maxsize=4)
def get_encoder(model_name: str, device: str = "cpu", max_length: int = 512) -> HuggingFaceEncoder:
    """Return a cached encoder instance to avoid repeated model loads."""

    return HuggingFaceEncoder(model_name=model_name, device=device, max_length=max_length)


def batch_iterable(items: Iterable[tuple[str, dict]], batch_size: int) -> Iterable[list[tuple[str, dict]]]:
    """Utility generator that groups chunk streams into batches."""

    batch: list[tuple[str, dict]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

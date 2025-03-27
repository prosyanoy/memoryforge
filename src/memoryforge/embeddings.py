"""Embedders. `hash` needs no downloads; `fastembed` gives real semantic vectors (ONNX, local)."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tiny concept lexicon so the offline embedder bridges obvious paraphrases
# ("how should we reach them" ~ contact_channel). Real semantics: use fastembed.
_CONCEPTS = {
    "reach": "contact", "contacted": "contact", "call": "phone contact", "calls": "phone contact",
    "ring": "phone", "mobile": "phone", "cell": "phone", "telephone": "phone",
    "mail": "email", "e": "email", "text": "sms", "texts": "sms", "message": "sms contact",
    "problem": "issue", "problems": "issue", "trouble": "issue", "broken": "issue", "complaint": "issue",
    "bug": "issue", "crash": "app_stability issue", "crashes": "app_stability issue",
    "charged": "billing", "charge": "billing", "refund": "billing", "invoice": "billing", "payment": "billing",
    "package": "shipping", "delivery": "shipping", "parcel": "shipping", "order": "shipping",
    "signin": "login", "password": "login", "locked": "login",
    "like": "preference likes", "likes": "preference likes", "love": "preference likes",
    "prefers": "preference", "prefer": "preference", "favorite": "preference likes",
    "hate": "dislikes preference", "dislike": "dislikes preference",
    "when": "time", "morning": "time", "evening": "time", "afternoon": "time",
    "speaks": "language", "speak": "language",
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class HashingEmbedder:
    """Signed feature hashing over word unigrams/bigrams, char trigrams and concept tokens."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    @staticmethod
    @lru_cache(maxsize=65536)
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % dim, 1.0 if (h >> 63) & 1 else -1.0

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = _TOKEN_RE.findall(text.lower())
        feats: list[tuple[str, float]] = []
        for w in words:
            feats.append((f"w:{w}", 1.0))
            for c in _CONCEPTS.get(w, "").split():
                feats.append((f"w:{c}", 0.8))
            padded = f"#{w}#"
            feats.extend((f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2))
        feats.extend((f"b:{a}_{b}", 0.5) for a, b in zip(words, words[1:]))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for feat, weight in self._features(t):
                idx, sign = self._bucket(feat, self.dim)
                out[i, idx] += sign * weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class FastEmbedEmbedder:
    def __init__(self, model: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the 'embeddings' extra: pip install -e '.[embeddings]'") from exc
        self._model = TextEmbedding(model)
        self.name = f"fastembed:{model}"
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fastembed":
        return FastEmbedEmbedder(settings.fastembed_model)
    return HashingEmbedder(settings.embedding_dim)

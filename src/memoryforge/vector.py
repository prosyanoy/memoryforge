"""Per-customer vector indexes (FAISS flat inner-product, with a NumPy fallback).

Every customer owns a separate partition, so a similarity search *cannot* return another
customer's vectors: there is no shared index to filter. Vectors are L2-normalised, so
inner product == cosine similarity.
"""

from __future__ import annotations

import threading
from typing import Iterable, Protocol

import numpy as np

try:  # pragma: no cover - exercised implicitly depending on environment
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class _Partition(Protocol):
    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None: ...
    def remove(self, ids: np.ndarray) -> None: ...
    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]: ...
    def __len__(self) -> int: ...


class _NumpyPartition:
    def __init__(self, dim: int):
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        self.ids = np.concatenate([self.ids, ids])
        self.mat = np.vstack([self.mat, vecs])

    def remove(self, ids: np.ndarray) -> None:
        keep = ~np.isin(self.ids, ids)
        self.ids, self.mat = self.ids[keep], self.mat[keep]

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.ids):
            return []
        sims = self.mat @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return [(int(self.ids[i]), float(sims[i])) for i in top]

    def __len__(self) -> int:
        return len(self.ids)


class _FaissPartition:
    def __init__(self, dim: int):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        self.index.add_with_ids(np.ascontiguousarray(vecs), ids)

    def remove(self, ids: np.ndarray) -> None:
        self.index.remove_ids(ids)

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        n = self.index.ntotal
        if not n:
            return []
        sims, ids = self.index.search(q[None, :], min(k, n))
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i != -1]

    def __len__(self) -> int:
        return int(self.index.ntotal)


class VectorIndex:
    def __init__(self, dim: int, backend: str = "auto"):
        if backend == "faiss" and faiss is None:
            raise RuntimeError("faiss-cpu is not installed")
        self.dim = dim
        self.backend = "faiss" if backend in ("auto", "faiss") and faiss is not None else "numpy"
        self._parts: dict[str, _Partition] = {}
        self._lock = threading.RLock()

    def _new(self) -> _Partition:
        return _FaissPartition(self.dim) if self.backend == "faiss" else _NumpyPartition(self.dim)

    def add(self, customer_id: str, memory_id: int, vec: np.ndarray) -> None:
        self.add_many(customer_id, [memory_id], np.asarray(vec, dtype=np.float32)[None, :])

    def add_many(self, customer_id: str, ids: Iterable[int], vecs: np.ndarray) -> None:
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        if not len(ids_arr):
            return
        with self._lock:
            self._parts.setdefault(customer_id, self._new()).add(ids_arr, np.asarray(vecs, dtype=np.float32))

    def remove(self, customer_id: str, ids: Iterable[int]) -> None:
        with self._lock:
            part = self._parts.get(customer_id)
            if part is not None:
                part.remove(np.asarray(list(ids), dtype=np.int64))

    def drop_customer(self, customer_id: str) -> None:
        with self._lock:
            self._parts.pop(customer_id, None)

    def clear(self) -> None:
        with self._lock:
            self._parts.clear()

    def search(self, customer_id: str, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (memory_id, cosine similarity) within ONE customer's partition."""
        with self._lock:
            part = self._parts.get(customer_id)
            return part.search(np.asarray(query, dtype=np.float32), k) if part else []

    def size(self, customer_id: str | None = None) -> int:
        with self._lock:
            if customer_id is not None:
                return len(self._parts.get(customer_id, ()))
            return sum(len(p) for p in self._parts.values())

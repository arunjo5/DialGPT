"""In-memory vector store with brute-force cosine search.

Fine for a single document's worth of chunks; swap for pgvector or Pinecone to scale.
"""
from __future__ import annotations

import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[str, list[float]]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, text: str, embedding: list[float]) -> None:
        self._items.append((text, embedding))

    def search(self, query: list[float], k: int) -> list[tuple[float, str]]:
        scored = [(_cosine(query, vec), text) for text, vec in self._items]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]

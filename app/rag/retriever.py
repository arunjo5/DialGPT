"""Document retriever: ingest text into the store, retrieve top-k chunks for a query."""
from __future__ import annotations

from app.rag.chunking import chunk_text
from app.rag.embedder import Embedder
from app.rag.store import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        *,
        top_k: int = 4,
        max_chars: int = 1200,
        overlap: int = 200,
    ) -> None:
        self._embedder = embedder
        self._store = VectorStore()
        self._top_k = top_k
        self._max_chars = max_chars
        self._overlap = overlap

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    async def ingest(self, text: str) -> int:
        chunks = chunk_text(text, max_chars=self._max_chars, overlap=self._overlap)
        if not chunks:
            return 0
        vectors = await self._embedder.embed(chunks)
        for chunk, vector in zip(chunks, vectors):
            self._store.add(chunk, vector)
        return len(chunks)

    async def search(self, query: str, k: int | None = None) -> list[str]:
        if not self._store:
            return []
        vectors = await self._embedder.embed([query])
        results = self._store.search(vectors[0], k or self._top_k)
        return [text for _, text in results]

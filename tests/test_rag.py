"""Retrieval engine tests: chunking, cosine search, and retrieval with a fake embedder."""
import pytest

from app.rag.chunking import chunk_text
from app.rag.retriever import Retriever
from app.rag.store import VectorStore


def test_chunk_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text_is_one_chunk():
    assert chunk_text("hello world", max_chars=100, overlap=10) == ["hello world"]


def test_chunks_slide_with_overlap():
    text = "abcdefghij" * 5  # 50 chars
    chunks = chunk_text(text, max_chars=20, overlap=5)
    assert chunks == [text[0:20], text[15:35], text[30:50]]


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=10, overlap=10)


def test_vector_store_ranks_by_cosine():
    store = VectorStore()
    store.add("parallel", [1.0, 0.0])
    store.add("orthogonal", [0.0, 1.0])
    store.add("diagonal", [1.0, 1.0])
    results = store.search([1.0, 0.0], k=3)
    assert [text for _, text in results] == ["parallel", "diagonal", "orthogonal"]


class FakeEmbedder:
    """Embeds text as keyword counts so retrieval is deterministic and offline."""

    VOCAB = ("cat", "dog", "bird", "fish")

    async def embed(self, texts):
        return [[float(t.lower().count(w)) for w in self.VOCAB] for t in texts]


async def test_retriever_returns_most_relevant_chunk():
    r = Retriever(FakeEmbedder(), top_k=1)
    await r.ingest("the cat sat on the mat")
    await r.ingest("the dog ran in the park")
    await r.ingest("a bird flew over the lake")
    assert r.chunk_count == 3
    top = await r.search("tell me about the cat")
    assert "cat" in top[0]


async def test_retriever_is_empty_until_ingested():
    r = Retriever(FakeEmbedder())
    assert await r.search("anything") == []

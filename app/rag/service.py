"""Process-wide handle to the document retriever, set once at startup."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.retriever import Retriever

_retriever: Retriever | None = None


def set_retriever(retriever: Retriever | None) -> None:
    global _retriever
    _retriever = retriever


def get_retriever() -> Retriever | None:
    return _retriever

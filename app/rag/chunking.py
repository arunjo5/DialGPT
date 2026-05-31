"""Split document text into overlapping chunks for embedding."""
from __future__ import annotations


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Sliding-window chunks of up to max_chars, each overlapping the last by `overlap`."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

"""Embedding via the OpenAI embeddings API."""
from __future__ import annotations

from typing import Protocol

import aiohttp

_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._api_key = api_key
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": texts},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]

"""Environment config and prompt assembly.

Importing this never fails on a missing key; OPENAI_API_KEY is validated at
startup (see main.py), so the module stays importable in tests. Domain code does
not import config.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from app.pdf import load_pdf_text

load_dotenv()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
PORT: int = int(os.getenv("PORT", "5050"))
MODEL: str = os.getenv("OPENAI_MODEL", "gpt-realtime")
VOICE: str = os.getenv("VOICE", "alloy")
PDF_PATH: str = os.getenv("PDF_PATH", "document.pdf")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# True: assistant speaks first. False: caller speaks first (legacy).
GREET_ON_CONNECT: bool = _env_flag("GREET_ON_CONNECT", True)

SYSTEM_MESSAGE: str = (
    "You are a helpful and adaptable AI assistant. Default language: English (en-US). "
    "Detect the language the user speaks in each turn and reply in that same language. "
    "If the user speaks in English reply in English; if they speak in Spanish reply in Spanish, etc. "
    "Do not switch languages mid-response; keep each spoken response entirely in a single language. "
    "If you are uncertain which language the user used, ask a short clarification question in the user's last-used language. "
    "Keep spoken responses uninterrupted unless the caller actually begins speaking."
)

GREETING_INSTRUCTION: str = (
    "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio "
    "and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can "
    "imagine. How can I help you?'"
)

PDF_TEXT: str = load_pdf_text(PDF_PATH)


def build_instructions(has_document: bool = False) -> str:
    instructions = SYSTEM_MESSAGE
    if has_document:
        instructions = (
            f"{instructions}\n\nThe caller has provided a document. Use the "
            "search_document tool to look up relevant passages before answering "
            "questions it might cover, and ground your answer in what it returns."
        )
    return instructions


def require_api_key() -> str:
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing the OpenAI API key. Please set OPENAI_API_KEY in the .env file."
        )
    return OPENAI_API_KEY

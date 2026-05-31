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


def build_instructions() -> str:
    instructions = SYSTEM_MESSAGE
    if PDF_TEXT:
        instructions = (
            f"{instructions}\n\nUse the following document as the primary source "
            f"when answering user questions:\n{PDF_TEXT}"
        )
    return instructions


def require_api_key() -> str:
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing the OpenAI API key. Please set OPENAI_API_KEY in the .env file."
        )
    return OPENAI_API_KEY

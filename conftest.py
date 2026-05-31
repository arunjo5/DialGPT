"""Pytest session setup.

Allow importing modules that read the API key at import time without a real
secret. The real fail-fast lives in app startup (main.py), not at import.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

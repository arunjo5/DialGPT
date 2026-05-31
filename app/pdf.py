"""PDF text extraction for optional grounding context."""
from __future__ import annotations


def load_pdf_text(path: str, max_chars: int = 20000) -> str:
    """Return up to max_chars of text from the PDF, or "" if it can't be read."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)[:max_chars]
    except Exception as e:
        print(f"Could not load PDF at {path}: {e}")
        return ""

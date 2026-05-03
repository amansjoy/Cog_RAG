"""PDF reading and sliding-window chunking.

The chunker keeps things simple: ~220 words per chunk with 30-word overlap.
For different document types you would tune these or use semantic splitting.
"""
from __future__ import annotations

import re

import pdfplumber


def read_pdf(path: str) -> str:
    """Concatenate text from all pages."""
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def chunk_text(text: str, target_words: int = 220, overlap_words: int = 30) -> list[str]:
    """Split text into overlapping word windows.

    Args:
        target_words: target chunk size in whitespace tokens.
        overlap_words: tokens shared between consecutive chunks.
    """
    tokens = [w for w in re.split(r"\s+", text) if w]
    chunks: list[str] = []
    i = 0
    step = target_words - overlap_words
    while i < len(tokens):
        window = tokens[i:i + target_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if i + target_words >= len(tokens):
            break
        i += step
    return chunks

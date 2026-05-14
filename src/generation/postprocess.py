"""Final answer post-processing."""
from __future__ import annotations


def clean_answer(text: str) -> str:
    return " ".join(text.split()).strip()

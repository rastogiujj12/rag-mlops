"""BM25 index persisted via pickle.

This module uses the standard `rank-bm25` BM25Okapi implementation while
keeping the project local-corpus-first. BEIR/SciFact dataset dependencies are
not required.
"""
from __future__ import annotations

import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunking.chunk import Chunk
from src.utils.logging import logger

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 and lightweight lexical guardrails."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    """Pickleable BM25 index plus the chunk metadata it scores against."""

    bm25: BM25Okapi
    chunk_ids: list[str]
    chunks: list[Chunk]

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> BM25Index:
        chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in chunks]
        if not chunks:
            raise ValueError("Cannot build BM25 index: no chunks were provided.")
        if not any(tokenized):
            raise ValueError("Cannot build BM25 index: all chunks tokenized to empty text.")
        bm25 = BM25Okapi(tokenized)
        logger.info(f"Built BM25 index over {len(chunks)} chunks")
        return cls(bm25=bm25, chunk_ids=[c.chunk_id for c in chunks], chunks=chunks)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved BM25 index → {path}")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            return pickle.load(f)

"""Retrieval base contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.chunking.chunk import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    source: str = ""  # "bm25" | "dense" | "hybrid"


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        ...

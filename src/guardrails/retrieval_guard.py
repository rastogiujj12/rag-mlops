"""Retrieval guard — runs *after* retrieval, before context assembly."""
from __future__ import annotations

from src.core.config import settings
from src.retrieval.base import RetrievalResult


def filter_results(
    results: list[RetrievalResult],
    min_score: float | None = None,
) -> list[RetrievalResult]:
    """Drop low-score hits and deduplicate by chunk_id (keeping the best-ranked occurrence)."""
    threshold = min_score if min_score is not None else settings.min_retrieval_score
    seen: set[str] = set()
    out: list[RetrievalResult] = []
    for r in results:
        if r.score < threshold:
            continue
        if r.chunk.chunk_id in seen:
            continue
        seen.add(r.chunk.chunk_id)
        out.append(r)
    return out


def has_minimum_quality(results: list[RetrievalResult], min_count: int = 1) -> bool:
    return len(results) >= min_count

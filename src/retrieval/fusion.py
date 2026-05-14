"""Fusion strategies. Currently RRF; reranking by lexical overlap as a tie-breaker.

RRF reference: Cormack, Clarke & Buettcher (SIGIR '09). It is known to be
robust and parameter-light — preferred over score normalization across
retrievers with incomparable score scales (BM25 vs cosine).
"""
from __future__ import annotations

from collections import defaultdict

from src.indexing.bm25 import tokenize
from src.retrieval.base import RetrievalResult


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse multiple ranked lists. Returns one merged list ordered by RRF score."""
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievalResult] = {}
    for ranking in rankings:
        for r in ranking:
            cid = r.chunk.chunk_id
            fused[cid] += 1.0 / (k + r.rank + 1)
            # keep the higher-ranked occurrence's metadata
            if cid not in by_id or r.rank < by_id[cid].rank:
                by_id[cid] = r

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievalResult(
            chunk=by_id[cid].chunk,
            score=score,
            rank=rank,
            source="hybrid",
        )
        for rank, (cid, score) in enumerate(ordered)
    ]


def lexical_overlap_rerank(
    query: str,
    results: list[RetrievalResult],
    weight: float = 0.3,
) -> list[RetrievalResult]:
    """Boost results whose tokens overlap heavily with the query.

    Mixes the existing fusion score with a lexical-overlap signal:
        new_score = (1-w) * score_norm + w * overlap
    """
    q_tokens = set(tokenize(query))
    if not q_tokens or not results:
        return results

    max_score = max(r.score for r in results) or 1.0
    rescored = []
    for r in results:
        c_tokens = set(tokenize(r.chunk.text))
        overlap = len(q_tokens & c_tokens) / len(q_tokens)
        new_score = (1 - weight) * (r.score / max_score) + weight * overlap
        rescored.append((new_score, r))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [
        RetrievalResult(chunk=r.chunk, score=s, rank=i, source=r.source)
        for i, (s, r) in enumerate(rescored)
    ]

"""IR evaluation metrics. Pure functions, fully unit-testable."""
from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / k


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for i, d in enumerate(retrieved_doc_ids, start=1):
        if d in relevant_doc_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Standard nDCG@k with binary or graded relevance.

    relevance: doc_id -> graded relevance (e.g., 0/1/2)
    """
    dcg = 0.0
    for i, d in enumerate(retrieved_doc_ids[:k], start=1):
        rel = relevance.get(d, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 1) for i, r in enumerate(ideal_rels, start=1) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

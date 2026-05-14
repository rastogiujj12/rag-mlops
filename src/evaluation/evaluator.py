"""Evaluator. Runs a retriever against (queries, qrels) and computes IR metrics."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from src.core.config import settings
from src.evaluation.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.base import Retriever
from src.utils.logging import logger


def _to_doc_ids(results) -> list[str]:
    """Map chunk-level results back to *document* level for IR evaluation.

    Multiple chunks of the same doc collapse to the doc's first occurrence.
    """
    seen, out = set(), []
    for r in results:
        if r.chunk.doc_id not in seen:
            out.append(r.chunk.doc_id)
            seen.add(r.chunk.doc_id)
    return out


def evaluate_retriever(
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: Iterable[int] | None = None,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Evaluate a retriever; returns a flat dict of mean metrics."""
    k_values = tuple(k_values) if k_values else settings.eval_k_values
    top_k_eval = max(max(k_values), 10)

    items = list(queries.items())
    if max_queries:
        items = items[:max_queries]

    per_q: dict[str, list[float]] = defaultdict(list)
    for qid, qtext in items:
        relevance = qrels.get(qid, {})
        relevant = {d for d, r in relevance.items() if r > 0}
        if not relevant:
            continue

        retrieved = retriever.retrieve(qtext, top_k_eval)
        doc_ids = _to_doc_ids(retrieved)

        for k in k_values:
            per_q[f"recall@{k}"].append(recall_at_k(doc_ids, relevant, k))
            per_q[f"precision@{k}"].append(precision_at_k(doc_ids, relevant, k))
            per_q[f"ndcg@{k}"].append(ndcg_at_k(doc_ids, relevance, k))
        per_q["mrr"].append(reciprocal_rank(doc_ids, relevant))

    metrics = {name: aggregate(vals) for name, vals in per_q.items()}
    metrics["num_queries"] = float(len(per_q.get("mrr", [])))
    logger.info(f"Evaluation done over {int(metrics['num_queries'])} queries")
    return metrics

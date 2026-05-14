"""Evaluation pipeline for local PDF/JSONL corpora.

The project intentionally avoids bundled benchmark datasets. Evaluation is driven
by a small local file at data/evaluation/eval_questions.json.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.chunking.chunk import Chunk
from src.core.config import settings
from src.evaluation.benchmark import benchmark_all
from src.guardrails.pipeline_guard import validate_metrics_threshold
from src.indexing.bm25 import BM25Index
from src.pipeline.ingest_pipeline import BM25_PATH, load_chunks, load_qrels, load_queries
from src.utils.logging import logger


def _chunk_id_to_doc_id(chunks: list[Chunk]) -> dict[str, str]:
    return {chunk.chunk_id: chunk.doc_id for chunk in chunks}


def _load_local_eval_dataset(chunks: list[Chunk]) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Load local retrieval-evaluation questions.

    Expected format:

    [
      {
        "id": "q1",
        "question": "What limitations are discussed?",
        "relevant_doc_ids": ["paper_a"],
        "relevant_chunk_ids": ["paper_a_chunk_000"]
      }
    ]

    Either relevant_doc_ids or relevant_chunk_ids may be used. Chunk ids are
    mapped back to doc ids because the current metrics are document-level.
    """
    path = settings.local_eval_path
    if not path.exists():
        return {}, {}

    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("questions") or raw.get("items") or []
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported evaluation dataset format in {path}")

    chunk_to_doc = _chunk_id_to_doc_id(chunks)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(items, start=1):
        qid = str(item.get("id") or item.get("qid") or f"q{idx:03d}")
        question = item.get("question") or item.get("query")
        if not question:
            continue

        relevant_docs = {str(doc_id) for doc_id in item.get("relevant_doc_ids", [])}
        for chunk_id in item.get("relevant_chunk_ids", []):
            doc_id = chunk_to_doc.get(str(chunk_id))
            if doc_id:
                relevant_docs.add(doc_id)

        if not relevant_docs:
            continue

        queries[qid] = str(question)
        qrels[qid] = {doc_id: 1 for doc_id in sorted(relevant_docs)}

    logger.info(f"Loaded {len(queries)} local evaluation questions from {path}")
    return queries, qrels


def _write_eval_report(summary: dict) -> None:
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    report_json = settings.eval_dir / "evaluation_report.json"
    report_md = settings.eval_dir / "evaluation_report.md"
    best = summary["best_model"]
    results = summary["results"]
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "best_model": best,
        "primary_metric": settings.primary_metric,
        "results": results,
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Created: {payload['created_at_utc']}",
        f"Best dense model: `{best}`",
        f"Primary metric: `{settings.primary_metric}`",
        "",
        "| Retriever | MRR | Recall@5 | Recall@10 | Precision@5 | nDCG@10 | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(results.items()):
        lines.append(
            f"| {name} | {metrics.get('mrr', 0):.4f} | {metrics.get('recall@5', 0):.4f} | "
            f"{metrics.get('recall@10', 0):.4f} | {metrics.get('precision@5', 0):.4f} | "
            f"{metrics.get('ndcg@10', 0):.4f} | {metrics.get('num_queries', 0):.0f} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote evaluation report → {report_md}")


def run_evaluate_pipeline(max_queries: int | None = None) -> dict:
    chunks = load_chunks()
    queries = load_queries()
    qrels = load_qrels()

    if not queries or not qrels:
        queries, qrels = _load_local_eval_dataset(chunks)

    if not queries or not qrels:
        raise RuntimeError(
            "No local evaluation dataset found. Add data/evaluation/eval_questions.json with "
            "question + relevant_doc_ids/relevant_chunk_ids before running retrieval evaluation."
        )

    bm25 = BM25Index.load(BM25_PATH)
    summary = benchmark_all(chunks, queries, qrels, bm25, max_queries=max_queries)

    best = summary["best_model"]
    dense_metrics = summary["results"][f"dense:{best}"]
    validate_metrics_threshold(dense_metrics, min_value=settings.min_eval_mrr)
    _write_eval_report(summary)
    logger.info(f"Evaluation pipeline complete. Best={best}")
    return summary

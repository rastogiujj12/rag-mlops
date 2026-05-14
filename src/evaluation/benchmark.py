"""Benchmarks every embedding model in the registry + BM25 + hybrid, then selects the best dense model."""
from __future__ import annotations

import json

from src.chunking.chunk import Chunk
from src.core.config import settings
from src.core.registry import EMBEDDING_REGISTRY, save_best_model
from src.evaluation.evaluator import evaluate_retriever
from src.indexing.bm25 import BM25Index
from src.indexing.dense import DenseIndex, dense_index_path
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.utils.logging import logger


def _load_or_build_dense(chunks: list[Chunk], model_name: str) -> DenseIndex:
    spec = EMBEDDING_REGISTRY[model_name]
    path = dense_index_path(model_name)
    if (path / "vectors.faiss").exists():
        logger.info(f"Loading cached dense index: {model_name}")
        return DenseIndex.load(path, spec)
    idx = DenseIndex.build(chunks, spec)
    idx.save(path)
    return idx


def benchmark_all(
    chunks: list[Chunk],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    bm25_index: BM25Index,
    max_queries: int | None = None,
) -> dict:
    """Run every retriever variant, write a results JSON, and pick a winner.

    Returns the full results dict.
    """
    results: dict[str, dict] = {}

    # 1. BM25 baseline
    logger.info("=== Evaluating BM25 ===")
    bm25_retriever = BM25Retriever(bm25_index)
    results["bm25"] = evaluate_retriever(bm25_retriever, queries, qrels, max_queries=max_queries)

    # 2. Each dense model
    dense_indices: dict[str, DenseIndex] = {}
    for name in EMBEDDING_REGISTRY:
        logger.info(f"=== Evaluating dense: {name} ===")
        idx = _load_or_build_dense(chunks, name)
        dense_indices[name] = idx
        retriever = DenseRetriever(idx)
        results[f"dense:{name}"] = evaluate_retriever(retriever, queries, qrels, max_queries=max_queries)

    # 3. Pick best dense model by primary metric
    primary = settings.primary_metric
    dense_scores = {n: results[f"dense:{n}"].get(primary, 0.0) for n in EMBEDDING_REGISTRY}
    best_name = max(dense_scores, key=dense_scores.get)
    logger.info(f"Best dense model by {primary}: {best_name} ({dense_scores[best_name]:.4f})")

    # 4. Hybrid using best dense + BM25
    logger.info(f"=== Evaluating hybrid (BM25 + {best_name}) ===")
    hybrid = HybridRetriever(
        bm25=bm25_retriever,
        dense=DenseRetriever(dense_indices[best_name]),
    )
    results[f"hybrid:bm25+{best_name}"] = evaluate_retriever(hybrid, queries, qrels, max_queries=max_queries)

    # 5. Persist results
    out_path = settings.eval_dir / "benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote benchmark → {out_path}")

    # 6. Save best-model registry record (uses dense metrics — that's what we are selecting)
    save_best_model(best_name, results[f"dense:{best_name}"])
    logger.info(f"Persisted best model → {best_name}")

    return {"results": results, "best_model": best_name}

"""End-to-end orchestration: ingest → index dense → evaluate → version artefacts."""
from __future__ import annotations

import os

from src.artifacts.versioning import copy_if_exists, new_run_id, write_metadata
from src.core.config import settings
from src.core.registry import load_best_model
from src.pipeline.evaluate_pipeline import run_evaluate_pipeline
from src.pipeline.ingest_pipeline import BM25_PATH, CHUNKS_PATH, CORPUS_PATH, run_ingest_pipeline
from src.pipeline.train_pipeline import run_train_pipeline
from src.utils.logging import logger


def _version_artifacts(run_id: str, eval_summary: dict | None, ingest_summary: dict, train_summary: dict) -> None:
    run_dir = settings.artifact_dir / run_id
    copy_if_exists(CORPUS_PATH, run_dir)
    copy_if_exists(CHUNKS_PATH, run_dir)
    copy_if_exists(BM25_PATH, run_dir)
    copy_if_exists(settings.index_dir / "dense", run_dir)
    copy_if_exists(settings.eval_dir / "benchmark.json", run_dir)
    copy_if_exists(settings.eval_dir / "evaluation_report.json", run_dir)
    copy_if_exists(settings.eval_dir / "evaluation_report.md", run_dir)
    copy_if_exists(settings.output_dir / "best_model.json", run_dir)

    write_metadata(
        run_dir,
        {
            "run_id": run_id,
            "embedding_model": (load_best_model() or {}).get("best_model"),
            "generation_provider": settings.llm_provider,
            "generation_model": settings.llm_model,
            "ingestion": ingest_summary,
            "training": train_summary,
            "evaluation": None if eval_summary is None else {"best_model": eval_summary["best_model"]},
        },
    )
    logger.info(f"Versioned artefacts → {run_dir}")


def run_full_pipeline(
    source: str = "pdf",
    max_eval_queries: int | None = None,
    skip_evaluation: bool = False,
    **ingest_kwargs,
) -> dict:
    logger.info("==== FULL PIPELINE START ====")
    ingest_summary = run_ingest_pipeline(source=source, **ingest_kwargs)
    train_summary = run_train_pipeline()
    eval_summary = None if skip_evaluation else run_evaluate_pipeline(max_queries=max_eval_queries)
    run_id = new_run_id(os.getenv("GITHUB_SHA"))
    _version_artifacts(run_id, eval_summary, ingest_summary, train_summary)
    logger.info("==== FULL PIPELINE DONE ====")
    return {
        "run_id": run_id,
        "ingestion": ingest_summary,
        "training": train_summary,
        "evaluation": None if eval_summary is None else {"best_model": eval_summary["best_model"]},
    }

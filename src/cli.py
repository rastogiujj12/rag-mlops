"""Command-line interface for local corpus RAG-MLOps workflows."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from src.core.config import settings
from src.pipeline.evaluate_pipeline import run_evaluate_pipeline
from src.pipeline.full_pipeline import run_full_pipeline
from src.pipeline.ingest_pipeline import run_ingest_pipeline
from src.pipeline.train_pipeline import run_train_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ingest(
    source: str = typer.Option("pdf", help="pdf | jsonl"),
    pdf_dir: Path = typer.Option(settings.raw_pdf_dir, help="Directory containing local PDFs"),
    jsonl_path: Path | None = typer.Option(None, help="Path to JSONL corpus when source=jsonl"),
):
    """Ingest a local corpus, chunk it, and build the BM25 index."""
    kwargs = {"pdf_dir": pdf_dir} if source == "pdf" else {"jsonl_path": jsonl_path}
    summary = run_ingest_pipeline(source=source, **kwargs)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def train(force: bool = typer.Option(False, help="Rebuild dense/FAISS indices even if cached")):
    """Build dense FAISS indices for all registered embedding models."""
    summary = run_train_pipeline(force=force)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def evaluate(max_queries: int | None = typer.Option(None)):
    """Evaluate retrieval against local evaluation questions."""
    summary = run_evaluate_pipeline(max_queries=max_queries)
    typer.echo(json.dumps({"best_model": summary["best_model"]}, indent=2))


@app.command("run-full")
def run_full(
    source: str = typer.Option("pdf", help="pdf | jsonl"),
    pdf_dir: Path = typer.Option(settings.raw_pdf_dir, help="Directory containing local PDFs"),
    jsonl_path: Path | None = typer.Option(None, help="Path to JSONL corpus when source=jsonl"),
    max_queries: int | None = typer.Option(None),
    skip_evaluation: bool = typer.Option(False, help="Build/version indices without running retrieval evaluation"),
):
    """Run ingestion → FAISS indexing → optional evaluation → artefact versioning."""
    kwargs = {"pdf_dir": pdf_dir} if source == "pdf" else {"jsonl_path": jsonl_path}
    summary = run_full_pipeline(
        source=source,
        max_eval_queries=max_queries,
        skip_evaluation=skip_evaluation,
        **kwargs,
    )
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()

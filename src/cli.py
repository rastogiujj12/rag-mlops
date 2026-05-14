"""Command-line interface for local corpus RAG-MLOps workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.pipeline.evaluate_pipeline import run_evaluate_pipeline
from src.pipeline.full_pipeline import run_full_pipeline
from src.pipeline.ingest_pipeline import run_ingest_pipeline
from src.pipeline.train_pipeline import run_train_pipeline


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["pdf", "jsonl"], default="pdf")
    parser.add_argument("--pdf-dir", type=Path, default=settings.raw_pdf_dir)
    parser.add_argument("--jsonl-path", type=Path, default=None)


def _corpus_kwargs(args: argparse.Namespace) -> dict[str, Path | None]:
    if args.source == "pdf":
        return {"pdf_dir": args.pdf_dir}
    if args.jsonl_path is None:
        raise SystemExit("--jsonl-path is required when --source jsonl")
    return {"jsonl_path": args.jsonl_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Local-corpus RAG-MLOps command line workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    _add_corpus_args(ingest_parser)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--force", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--max-queries", type=int, default=None)

    run_full_parser = subparsers.add_parser("run-full")
    _add_corpus_args(run_full_parser)
    run_full_parser.add_argument("--max-queries", type=int, default=None)
    run_full_parser.add_argument("--skip-evaluation", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        summary = run_ingest_pipeline(source=args.source, **_corpus_kwargs(args))
        _print_json(summary)
        return

    if args.command == "train":
        summary = run_train_pipeline(force=args.force)
        _print_json(summary)
        return

    if args.command == "evaluate":
        summary = run_evaluate_pipeline(max_queries=args.max_queries)
        _print_json({"best_model": summary["best_model"]})
        return

    if args.command == "run-full":
        summary = run_full_pipeline(
            source=args.source,
            max_eval_queries=args.max_queries,
            skip_evaluation=args.skip_evaluation,
            **_corpus_kwargs(args),
        )
        _print_json(summary)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
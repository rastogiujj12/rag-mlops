"""Pipeline-level guards. Validates artifacts and prevents bad models from being promoted."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.config import settings


class PipelineGuardError(Exception):
    pass


def validate_corpus(docs) -> None:
    if not docs:
        raise PipelineGuardError("Corpus is empty")
    null_text = sum(1 for d in docs if not (d.text or "").strip())
    if null_text / len(docs) > 0.10:
        raise PipelineGuardError(f"Too many empty documents: {null_text}/{len(docs)}")


def validate_chunks(chunks) -> None:
    if not chunks:
        raise PipelineGuardError("No chunks produced")
    avg_len = sum(len(c.text.split()) for c in chunks) / len(chunks)
    if avg_len < 5:
        raise PipelineGuardError(f"Chunks too short on average: {avg_len:.1f} tokens")


def validate_index(index_path: Path) -> None:
    if not index_path.exists():
        raise PipelineGuardError(f"Missing index artifact: {index_path}")


def validate_metrics_threshold(
    metrics: dict,
    metric: str | None = None,
    min_value: float = 0.10,
) -> None:
    """Enforce a floor on the chosen metric. Used in CT to prevent promoting a regressed model."""
    metric = metric or settings.primary_metric
    value = metrics.get(metric, 0.0)
    if value < min_value:
        raise PipelineGuardError(
            f"Quality gate failed: {metric}={value:.4f} below minimum {min_value:.4f}"
        )


def validate_no_regression(
    new_metrics: dict,
    history_path: Path | None = None,
    metric: str | None = None,
    tolerance: float = 0.02,
) -> None:
    """Refuse to deploy if the new model is materially worse than the previous one."""
    metric = metric or settings.primary_metric
    history_path = history_path or (settings.eval_dir / "history.json")
    if not history_path.exists():
        # First run — record and exit
        history_path.write_text(json.dumps([{metric: new_metrics.get(metric, 0.0)}]))
        return
    history = json.loads(history_path.read_text())
    prev = history[-1].get(metric, 0.0) if history else 0.0
    current = new_metrics.get(metric, 0.0)
    if current + tolerance < prev:
        raise PipelineGuardError(
            f"Regression detected on {metric}: {current:.4f} < previous {prev:.4f} - {tolerance}"
        )
    history.append({metric: current})
    history_path.write_text(json.dumps(history, indent=2))

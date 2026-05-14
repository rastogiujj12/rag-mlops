"""Lightweight artefact versioning utilities.

This intentionally avoids heavy MLflow-style registry complexity. Each training
run can be associated with a corpus/config hash, Git commit, metrics and the
paths to generated indices.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import settings


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        if not path.exists() or not path.is_file():
            continue
        h.update(str(path.relative_to(settings.project_root) if path.is_relative_to(settings.project_root) else path).encode())
        h.update(sha256_file(path).encode())
    return h.hexdigest()


def corpus_hash() -> str | None:
    candidates = []
    for rel in ["data/raw", "data/processed"]:
        root = settings.project_root / rel
        if root.exists():
            candidates.extend(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep")
    if not candidates:
        return None
    return hash_paths(candidates)


def new_run_id(commit: str | None = None) -> str:
    commit = commit or os.getenv("GITHUB_SHA") or settings.git_commit or "local"
    short = commit[:7]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{short}"


def write_metadata(run_dir: Path, payload: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": os.getenv("GITHUB_SHA") or settings.git_commit,
        "docker_image_tag": settings.docker_image_tag,
        "workspace": settings.workspace,
        "corpus_hash": corpus_hash(),
        **payload,
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def copy_if_exists(source: Path, destination_dir: Path) -> None:
    if not source.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def latest_run_metadata() -> dict[str, Any] | None:
    if not settings.artifact_dir.exists():
        return None
    candidates = sorted(settings.artifact_dir.glob("*/metadata.json"), reverse=True)
    if not candidates:
        return None
    return json.loads(candidates[0].read_text(encoding="utf-8"))

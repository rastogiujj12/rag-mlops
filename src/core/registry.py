"""Embedding model registry.

The whole point: never hardcode a model name in business logic. New models
are added here; the rest of the system (indexing, retrieval, evaluation)
iterates over this registry.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from src.core.config import settings


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str  # short id used in filenames / output keys
    hf_id: str  # HuggingFace repo id
    dim: int
    query_prefix: str = ""  # some models (e5) need "query: " / "passage: " prefixes
    passage_prefix: str = ""
    max_seq_length: int = 512


# Add new models here. Nothing else changes.
EMBEDDING_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "e5-base": EmbeddingModelSpec(
        name="e5-base",
        hf_id="intfloat/e5-base-v2",
        dim=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "bge-small": EmbeddingModelSpec(
        name="bge-small",
        hf_id="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    ),
    "minilm": EmbeddingModelSpec(
        name="minilm",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
    ),
}


# ----- Best-model registry (written by evaluation pipeline) -----

BEST_MODEL_PATH = settings.output_dir / "best_model.json"


def save_best_model(name: str, metrics: dict) -> None:
    spec = EMBEDDING_REGISTRY[name]
    payload = {
        "best_model": name,
        "spec": asdict(spec),
        "metrics": metrics,
    }
    BEST_MODEL_PATH.write_text(json.dumps(payload, indent=2))


def load_best_model() -> dict | None:
    if not BEST_MODEL_PATH.exists():
        return None
    return json.loads(BEST_MODEL_PATH.read_text())


def get_best_model_spec() -> EmbeddingModelSpec:
    """Returns the active dense model. Falls back to first registry entry if none selected yet."""
    record = load_best_model()
    if record and record["best_model"] in EMBEDDING_REGISTRY:
        return EMBEDDING_REGISTRY[record["best_model"]]
    # sensible default during cold-start
    return next(iter(EMBEDDING_REGISTRY.values()))

"""Thin facade over registry.get_best_model_spec for explicitness in pipelines."""
from __future__ import annotations

from src.core.registry import EmbeddingModelSpec, get_best_model_spec, load_best_model


def active_dense_model() -> EmbeddingModelSpec:
    return get_best_model_spec()


def selection_record() -> dict | None:
    return load_best_model()

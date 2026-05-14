"""Build dense indices for every registered embedding model.

In a true 'training' sense we don't fine-tune embeddings here, but the
indexing of frozen models *is* the training step in a retrieval-only RAG —
it's what produces the model artifacts the API serves.
"""
from __future__ import annotations

from src.core.registry import EMBEDDING_REGISTRY
from src.indexing.dense import DenseIndex, dense_index_path
from src.pipeline.ingest_pipeline import load_chunks
from src.utils.logging import logger


def run_train_pipeline(force: bool = False) -> dict:
    chunks = load_chunks()
    built = []
    for name, spec in EMBEDDING_REGISTRY.items():
        path = dense_index_path(name)
        if path.exists() and not force and (path / "vectors.faiss").exists():
            logger.info(f"Skipping {name} (already indexed)")
            continue
        idx = DenseIndex.build(chunks, spec)
        idx.save(path)
        built.append(name)
    return {"built_indices": built, "all_models": list(EMBEDDING_REGISTRY)}

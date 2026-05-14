"""SentenceTransformer model loading with a persistent cache.

The first run can download a configured Hugging Face/SentenceTransformers
model into a mounted cache directory. Later runs can load the same model from
that cache without contacting Hugging Face.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.config import settings
from src.utils.logging import logger


def load_sentence_transformer(
    model_name: str,
    *,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
) -> Any:
    """Load a SentenceTransformer using the shared model cache.

    Parameters
    ----------
    model_name:
        Hugging Face/SentenceTransformers model id, e.g. ``intfloat/e5-base-v2``.
    device:
        ``cpu`` or ``cuda``. Defaults to ``settings.embedding_device``.
    cache_dir:
        Directory mounted as a persistent volume. Defaults to
        ``settings.embedding_cache_dir``.
    offline:
        When true, fail if the model is not already present in the cache.
        When false, SentenceTransformers may download the model on first use.
    """
    resolved_cache_dir = Path(cache_dir or settings.embedding_cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = device or settings.embedding_device
    resolved_offline = settings.embedding_offline if offline is None else offline

    logger.info(
        f"Loading embedding model '{model_name}' from cache "
        f"'{resolved_cache_dir}' (device={resolved_device}, offline={resolved_offline})"
    )

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            model_name,
            device=resolved_device,
            cache_folder=str(resolved_cache_dir),
            local_files_only=resolved_offline,
        )
    except Exception as exc:
        if resolved_offline:
            raise RuntimeError(
                "Embedding model could not be loaded in offline mode. "
                f"Model: '{model_name}'. Cache directory: '{resolved_cache_dir}'. "
                "Run once with RAG_EMBEDDING_OFFLINE=false and internet access, "
                "or pre-populate/mount the model cache volume."
            ) from exc
        raise

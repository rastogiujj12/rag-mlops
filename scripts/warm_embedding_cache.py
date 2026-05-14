#!/usr/bin/env python3
"""Download/cache the configured SentenceTransformer models once.

Useful before offline demos or when preparing the mounted model-cache volume.
"""
from __future__ import annotations

from src.core.registry import EMBEDDING_REGISTRY
from src.indexing.model_loader import load_sentence_transformer


def main() -> None:
    for spec in EMBEDDING_REGISTRY.values():
        model = load_sentence_transformer(spec.hf_id, offline=False)
        model.max_seq_length = spec.max_seq_length
        print(f"Cached {spec.name}: {spec.hf_id}")


if __name__ == "__main__":
    main()

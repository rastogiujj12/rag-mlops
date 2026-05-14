"""Dense FAISS index. One index per embedding model (kept on disk separately)."""
from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import faiss
import numpy as np

from src.chunking.chunk import Chunk
from src.core.config import settings
from src.core.registry import EmbeddingModelSpec
from src.indexing.model_loader import load_sentence_transformer
from src.utils.logging import logger


@dataclass
class DenseIndex:
    _ENCODER_CACHE: ClassVar[dict[str, Any]] = {}

    spec: EmbeddingModelSpec
    index: faiss.Index
    chunk_ids: list[str]
    chunks: list[Chunk]

    @staticmethod
    def _encoder(spec: EmbeddingModelSpec) -> Any:
        # Cache avoids reloading the embedding model on every API query.
        if spec.name not in DenseIndex._ENCODER_CACHE:
            model = load_sentence_transformer(spec.hf_id)
            model.max_seq_length = spec.max_seq_length
            DenseIndex._ENCODER_CACHE[spec.name] = model
        return DenseIndex._ENCODER_CACHE[spec.name]

    @classmethod
    def build(cls, chunks: Iterable[Chunk], spec: EmbeddingModelSpec) -> DenseIndex:
        chunks = list(chunks)
        encoder = cls._encoder(spec)
        texts = [spec.passage_prefix + c.text for c in chunks]

        logger.info(f"Encoding {len(texts)} passages with {spec.name} ({spec.hf_id})…")
        emb = encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine sim via inner product
        ).astype(np.float32)

        index = faiss.IndexFlatIP(spec.dim)
        index.add(emb)
        logger.info(f"Built FAISS index ({emb.shape}) for {spec.name}")

        return cls(spec=spec, index=index, chunk_ids=[c.chunk_id for c in chunks], chunks=chunks)

    def encode_query(self, query: str) -> np.ndarray:
        encoder = self._encoder(self.spec)
        q = encoder.encode(
            [self.spec.query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return q

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dirpath / "vectors.faiss"))
        with (dirpath / "meta.pkl").open("wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "chunks": self.chunks}, f)
        (dirpath / "spec.json").write_text(json.dumps(self.spec.__dict__))
        logger.info(f"Saved dense index ({self.spec.name}) → {dirpath}")

    @classmethod
    def load(cls, dirpath: Path, spec: EmbeddingModelSpec) -> DenseIndex:
        index = faiss.read_index(str(dirpath / "vectors.faiss"))
        with (dirpath / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        return cls(spec=spec, index=index, chunk_ids=meta["chunk_ids"], chunks=meta["chunks"])


def dense_index_path(model_name: str) -> Path:
    return settings.index_dir / "dense" / model_name

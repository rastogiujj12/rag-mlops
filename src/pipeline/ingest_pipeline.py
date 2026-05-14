"""Ingestion → chunking → BM25 indexing. Persists artifacts to disk."""
from __future__ import annotations

import json
import pickle

from src.chunking.chunk import chunk_documents
from src.core.config import settings
from src.guardrails.pipeline_guard import validate_chunks, validate_corpus
from src.indexing.bm25 import BM25Index
from src.ingestion.ingest import ingest, save_corpus
from src.utils.logging import logger

CORPUS_PATH = settings.data_dir / "processed" / "corpus.jsonl"
CHUNKS_PATH = settings.data_dir / "processed" / "chunks.pkl"
QRELS_PATH = settings.data_dir / "processed" / "qrels.json"
QUERIES_PATH = settings.data_dir / "processed" / "queries.json"
BM25_PATH = settings.index_dir / "bm25.pkl"


def run_ingest_pipeline(source: str = "pdf", **kwargs) -> dict:
    logger.info(f"Running ingestion pipeline (source={source})")
    docs, queries, qrels = ingest(source=source, **kwargs)
    validate_corpus(docs)

    save_corpus(docs, CORPUS_PATH)

    chunks = chunk_documents(docs)
    validate_chunks(chunks)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks → {CHUNKS_PATH}")

    # These files may be populated later from local evaluation data. Remove stale
    # query/qrel files so old evaluations cannot accidentally gate a new corpus.
    if queries is not None:
        QUERIES_PATH.write_text(json.dumps(queries))
    elif QUERIES_PATH.exists():
        QUERIES_PATH.unlink()
    if qrels is not None:
        QRELS_PATH.write_text(json.dumps(qrels))
    elif QRELS_PATH.exists():
        QRELS_PATH.unlink()

    bm25_index = BM25Index.build(chunks)
    bm25_index.save(BM25_PATH)

    return {
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "has_qrels": qrels is not None,
        "source": source,
    }


def load_chunks() -> list:
    with CHUNKS_PATH.open("rb") as f:
        return pickle.load(f)


def load_queries() -> dict:
    return json.loads(QUERIES_PATH.read_text()) if QUERIES_PATH.exists() else {}


def load_qrels() -> dict:
    return json.loads(QRELS_PATH.read_text()) if QRELS_PATH.exists() else {}

"""FastAPI application — wires retriever + guardrails + LLM into the /query endpoint."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import (
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    Source,
    SourceDocumentResponse,
    VersionResponse,
)
from src.artifacts.versioning import corpus_hash, latest_run_metadata
from src.core.config import settings
from src.core.registry import get_best_model_spec, load_best_model
from src.generation.citations import assign_source_ids, build_references, validate_citations
from src.generation.llm import make_llm
from src.generation.postprocess import clean_answer
from src.generation.prompt import build_prompt
from src.guardrails.manager import post_generation, post_retrieval, pre_retrieval
from src.indexing.bm25 import BM25Index
from src.indexing.dense import DenseIndex, dense_index_path
from src.pipeline.ingest_pipeline import BM25_PATH
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.utils.logging import logger

STATE: dict = {}


def _load_retriever() -> HybridRetriever:
    bm25 = BM25Retriever(BM25Index.load(BM25_PATH))
    spec = get_best_model_spec()
    dense_idx = DenseIndex.load(dense_index_path(spec.name), spec)
    dense = DenseRetriever(dense_idx)
    return HybridRetriever(bm25=bm25, dense=dense)


def _source_models(source_dicts: list[dict]) -> list[Source]:
    return [Source(**source) for source in source_dicts]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        STATE["retriever"] = _load_retriever()
        STATE["best_model"] = get_best_model_spec().name
        logger.info(f"API ready. Active dense model: {STATE['best_model']}")
    except Exception as e:
        # Degrade gracefully — health endpoint will report not-ready
        logger.warning(f"Retriever not available at startup: {e}")
        STATE["retriever"] = None
        STATE["best_model"] = None
    yield


app = FastAPI(
    title="RAG-MLOps API",
    version="1.1.0",
    description="Hybrid retrieval-augmented generation over academic corpora",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if STATE.get("retriever") is not None else "no_index",
        best_model=STATE.get("best_model"),
    )


@app.get("/ready", response_model=HealthResponse)
def ready():
    if STATE.get("retriever") is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return HealthResponse(status="ready", best_model=STATE.get("best_model"))


@app.get("/version", response_model=VersionResponse)
def version():
    best = load_best_model() or {}
    latest = latest_run_metadata() or {}
    return VersionResponse(
        app_version=app.version,
        git_commit=settings.git_commit,
        docker_image_tag=settings.docker_image_tag,
        artifact_run_id=settings.active_artifact_run_id or latest.get("run_id"),
        corpus_hash=latest.get("corpus_hash") or corpus_hash(),
        embedding_model=(best.get("best_model") or STATE.get("best_model")),
        embedding_cache_dir=str(settings.embedding_cache_dir),
        embedding_offline=settings.embedding_offline,
        embedding_device=settings.embedding_device,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        ollama_host_configured=bool(settings.ollama_host),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    bench_path = settings.eval_dir / "benchmark.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
    return MetricsResponse(benchmark=bench, best_model=load_best_model())


@app.get("/sources/{doc_id}", response_model=SourceDocumentResponse)
def source_document(doc_id: str, limit: int = 20):
    retriever = STATE.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    # Use the BM25 index chunks as the canonical loaded chunk list.
    chunks = retriever.bm25.index.chunks
    matching = [chunk for chunk in chunks if chunk.doc_id == doc_id]
    if not matching:
        raise HTTPException(status_code=404, detail="Document not found")

    first = matching[0]
    sources = []
    for i, chunk in enumerate(matching[:limit], start=1):
        sources.append(
            Source(
                id=i,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                authors=chunk.authors,
                year=chunk.year,
                pages=None if not chunk.start_page else (f"p. {chunk.start_page}" if chunk.start_page == chunk.end_page else f"pp. {chunk.start_page}-{chunk.end_page}"),
                filename=chunk.filename,
                source_url=chunk.source_url,
                score=0.0,
                text=chunk.text[:500],
            )
        )
    return SourceDocumentResponse(
        doc_id=doc_id,
        title=first.title,
        authors=first.authors,
        year=first.year,
        filename=first.filename,
        source_url=first.source_url,
        chunk_count=len(matching),
        chunks=sources,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    retriever = STATE.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Index not loaded — run the ingest+train+evaluate pipeline first")

    pre = pre_retrieval(req.question)
    if not pre.proceed:
        return QueryResponse(
            answer="I cannot answer this query.",
            sources=[],
            references=[],
            fallback=True,
            fallback_reason=f"{pre.stage}: {pre.reason}",
        )

    raw = retriever.retrieve(req.question, req.top_k)

    post, filtered = post_retrieval(req.question, raw)
    source_dicts = assign_source_ids(filtered[: req.top_k])
    source_models = _source_models(source_dicts)
    references = build_references(source_dicts)

    if not post.proceed:
        return QueryResponse(
            answer="I cannot answer this from the available sources.",
            sources=source_models,
            references=references,
            coverage=post.coverage,
            fallback=True,
            fallback_reason=f"{post.stage}: {post.reason}",
        )

    system, user = build_prompt(req.question, filtered[: req.top_k])
    llm = make_llm(req.question, filtered[: req.top_k])
    answer = clean_answer(llm.generate(system, user))

    if not validate_citations(answer, len(source_dicts), require_at_least_one=True):
        return QueryResponse(
            answer="I cannot confidently answer this from the provided sources.",
            sources=source_models,
            references=references,
            coverage=post.coverage,
            fallback=True,
            fallback_reason="citation_guard: generated answer did not cite valid retrieved sources",
        )

    pg = post_generation(answer, filtered[: req.top_k])
    if not pg.proceed:
        return QueryResponse(
            answer="I cannot confidently answer this from the provided sources.",
            sources=source_models,
            references=references,
            coverage=post.coverage,
            grounding=pg.grounding,
            fallback=True,
            fallback_reason=f"{pg.stage}: {pg.reason}",
        )

    return QueryResponse(
        answer=answer,
        sources=source_models,
        references=references,
        coverage=post.coverage,
        grounding=pg.grounding,
        fallback=False,
    )

"""API request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class Source(BaseModel):
    id: int
    doc_id: str
    chunk_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    pages: str | None = None
    filename: str | None = None
    source_url: str | None = None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    references: list[str] = Field(default_factory=list)
    coverage: float | None = None
    grounding: float | None = None
    fallback: bool = False
    fallback_reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    best_model: str | None = None


class MetricsResponse(BaseModel):
    benchmark: dict | None = None
    best_model: dict | None = None


class VersionResponse(BaseModel):
    app_version: str
    git_commit: str
    docker_image_tag: str
    artifact_run_id: str | None = None
    corpus_hash: str | None = None
    embedding_model: str | None = None
    embedding_cache_dir: str
    embedding_offline: bool
    embedding_device: str
    llm_provider: str
    llm_model: str
    ollama_host_configured: bool


class SourceDocumentResponse(BaseModel):
    doc_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    filename: str | None = None
    source_url: str | None = None
    chunk_count: int
    chunks: list[Source]

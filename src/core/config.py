"""Centralized configuration. All knobs in one place — env-overridable."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    output_dir: Path = Path(__file__).resolve().parents[2] / "outputs"
    index_dir: Path = Path(__file__).resolve().parents[2] / "outputs" / "indices"
    eval_dir: Path = Path(__file__).resolve().parents[2] / "outputs" / "eval"
    artifact_dir: Path = Path(__file__).resolve().parents[2] / "artifacts" / "runs"

    # Build/version metadata. CI can set these.
    git_commit: str = "local"
    docker_image_tag: str = "local"
    active_artifact_run_id: str | None = None

    # Dataset/workspace
    raw_pdf_dir: Path = Path(__file__).resolve().parents[2] / "data" / "raw" / "pdfs"
    local_eval_path: Path = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "eval_questions.json"
    workspace: str = "default"

    # Chunking
    chunk_size: int = 256  # words/tokens approx by whitespace split
    chunk_overlap: int = 32

    # Embedding model cache/runtime
    # Mount this path as a persistent volume in Docker/Kubernetes.
    # First run can download into it; later runs can load offline from it.
    embedding_cache_dir: Path = Path("/models/sentence-transformers")
    embedding_device: str = "cpu"
    embedding_offline: bool = False

    # Retrieval
    top_k_bm25: int = 50
    top_k_dense: int = 50
    top_k_final: int = 10
    rrf_k: int = 60  # RRF damping constant

    # Evaluation
    eval_k_values: tuple[int, ...] = (1, 5, 10)
    primary_metric: Literal["mrr", "recall@10", "ndcg@10"] = "mrr"
    min_eval_mrr: float = 0.05

    # Guardrails
    min_retrieval_score: float = 0.05
    min_coverage_ratio: float = 0.15
    min_overlap_for_grounding: float = 0.20

    # LLM. Ollama is a runtime dependency; the pipeline does not train/deploy it.
    llm_provider: Literal["stub", "ollama", "openai", "anthropic"] = "stub"
    llm_model: str = "lokeshjothiram/rag-distiller-v1:4b"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    ollama_host: str = "http://host.docker.internal:11434"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()

# Ensure directories exist at import time
for _d in (settings.data_dir, settings.raw_pdf_dir, settings.local_eval_path.parent, settings.output_dir, settings.index_dir, settings.eval_dir, settings.artifact_dir):
    _d.mkdir(parents=True, exist_ok=True)

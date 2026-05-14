# RAG-MLOps: Production Academic RAG System

A modular, production-oriented Retrieval-Augmented Generation system for academic literature with hybrid retrieval, source-grounded citations, artefact versioning, guardrails, and CI/CT/CD automation.

## Highlights

- **Hybrid retrieval**: BM25 lexical retrieval + E5 embeddings + FAISS vector search, fused with Reciprocal Rank Fusion and lexical-overlap reranking.
- **Grounded generation**: FastAPI calls an LLM provider for answer generation; Ollama can be used locally as an external GPU runtime dependency.
- **Citation support**: retrieved chunks receive source markers like `[1]`, and the backend formats Harvard-style references from stored metadata.
- **Evaluation**: local evaluation questions with Recall@K, Precision@K, MRR, nDCG, and automatic evaluation reports.
- **MLOps**: GitHub Actions for CI, CT, Docker image publishing, deployment gates, and optional ArgoCD/Kind GitOps deployment.
- **Versioning**: retrieval artefacts are stored under `artifacts/runs/<timestamp>_<commit>/` with metadata linking the corpus, config, Docker image, and Git commit.

## What is “training” in this project?

The pipeline does **not** train or fine-tune an LLM. The continuous training stage rebuilds the retrieval artefacts used by the RAG service:

- cleaned corpus/chunks,
- BM25 index,
- E5 embeddings,
- FAISS vector index,
- evaluation report,
- active retrieval configuration.

Ollama is treated as an external runtime dependency for generation and is tracked in `/version`, but it is not trained or deployed by the pipeline.

## Architecture

```text
Local PDF corpus
      ↓
Ingestion & preprocessing
      ↓
Chunking + metadata extraction
      ↓
BM25 index + E5 embeddings + FAISS index
      ↓
Retrieval evaluation + quality gate
      ↓
Versioned artefact folder
      ↓
Docker image build
      ↓
ArgoCD or self-hosted runner deployment
      ↓
Kind Kubernetes cluster
      ↓
FastAPI RAG service
      ↓
Local Ollama runtime
```

## Dependency-aware retraining

The intended CT behaviour is dependency-aware:

| Change | Stages rerun |
|---|---|
| Corpus/PDF changes | ingestion → preprocessing → chunking → embedding → BM25 → FAISS → evaluation |
| Text cleaning changes | preprocessing → chunking → embedding → BM25 → FAISS → evaluation |
| Chunking changes | chunking → embedding → BM25 → FAISS → evaluation |
| Embedding model changes | embedding → FAISS → evaluation |
| Retrieval logic changes | retrieval tests → evaluation |
| API-only changes | tests → Docker build → deployment |
| Documentation-only changes | no retraining |

## API endpoints

- `GET /health` — liveness/status; returns `no_index` if artefacts are missing.
- `GET /ready` — readiness; returns HTTP 503 until retrieval artefacts are loaded.
- `GET /version` — Git commit, Docker image tag, active artefact run, corpus hash, embedding model, and Ollama model.
- `GET /metrics` — benchmark/evaluation results.
- `GET /sources/{doc_id}` — document metadata and chunk previews.
- `POST /query` — answer generation with source citations and references.

Example query:

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the limitations discussed in the paper?", "top_k": 5}'
```

## Local quickstart

This project is local-corpus-first. Put your PDFs in `data/raw/pdfs/`; the system does not download benchmark datasets.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add local PDFs
mkdir -p data/raw/pdfs
# copy your .pdf files into data/raw/pdfs/

# Optional: add citation metadata
# data/raw/pdfs/metadata.json

# 3. Build local artefacts without evaluation first
make local-run

# 4. Start API in stub mode
RAG_LLM_PROVIDER=stub uvicorn app.main:app --reload --port 8000

# 5. Check service
curl localhost:8000/health
curl localhost:8000/ready
curl localhost:8000/version
```

After you create `data/evaluation/eval_questions.json`, run the gated pipeline:

```bash
make pipeline
```

Example `data/evaluation/eval_questions.json`:

```json
[
  {
    "id": "q1",
    "question": "What are the main limitations discussed in the paper?",
    "relevant_doc_ids": ["paper1"]
  }
]
```

You can also use `relevant_chunk_ids` instead of `relevant_doc_ids` once you know the generated chunk IDs.


## Embedding model cache

SentenceTransformer models are loaded through a persistent cache volume. On the first run, if the configured model is not present, it can be downloaded from Hugging Face and stored in the cache. Later runs reuse the local cache and do not need to call Hugging Face again.

Key settings:

```bash
RAG_EMBEDDING_CACHE_DIR=/models/sentence-transformers
RAG_EMBEDDING_DEVICE=cpu
RAG_EMBEDDING_OFFLINE=false
```

Recommended flow:

```text
First setup/demo preparation: RAG_EMBEDDING_OFFLINE=false  # download/cache once
Offline demo/runtime:       RAG_EMBEDDING_OFFLINE=true   # load only from mounted cache
```

You can pre-warm the cache with:

```bash
python scripts/warm_embedding_cache.py
```

If `RAG_EMBEDDING_OFFLINE=true` and the model is not already cached, startup/training will fail with a clear error explaining how to populate the cache.

## Ollama mode

Run Ollama locally on the host/GPU machine, then start the API with:

```bash
RAG_LLM_PROVIDER=ollama \
RAG_LLM_MODEL=lokeshjothiram/rag-distiller-v1:4b \
RAG_OLLAMA_HOST=http://localhost:11434 \
uvicorn app.main:app --reload --port 8000
```

When running inside Kind/Docker, configure `RAG_OLLAMA_HOST` so the pod/container can reach the host Ollama service.

## Docker

```bash
docker build -t rag-mlops .
docker run -p 8000:8000 \
  -e RAG_EMBEDDING_CACHE_DIR=/models/sentence-transformers \
  -e RAG_EMBEDDING_OFFLINE=false \
  -v $(pwd)/models/sentence-transformers:/models/sentence-transformers \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/artifacts:/app/artifacts \
  rag-mlops
```

## Kind / ArgoCD deployment

Kubernetes manifests live in `k8s/`. ArgoCD application scaffolding lives in `argocd/`.

The deployment mounts a hostPath model cache at `/models/sentence-transformers` so the E5/SentenceTransformer model persists across pod restarts.

Demo commands:

```bash
# Prepare folders that will be mounted into the Kind node
./scripts/prepare_kind_runtime.sh

# Create Kind with extraMounts so host artefacts/cache are visible to pods
kind create cluster --config kind/cluster.yaml

# Deploy
kubectl apply -f k8s/
kubectl rollout status deployment/rag-api
kubectl port-forward svc/rag-api-service 8000:8000
./scripts/smoke_test.sh
```

Important: the Docker image intentionally contains application code and dependencies, not the generated retrieval artefacts. For local Kind, `outputs/` and `artifacts/` are mounted into the cluster through the Kind node using `kind/cluster.yaml`. This keeps code versions and retrieval artefact versions separate.

On Linux, `host.docker.internal` may not resolve inside Kind depending on Docker/network configuration. If the FastAPI pod cannot reach Ollama, run Ollama so it listens on the host network and set `RAG_OLLAMA_HOST` in `k8s/deployment.yaml` to your host IP, for example `http://192.168.x.x:11434`. Keep this restricted to your local network/firewall for the demo.

## CI / CT / CD

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | every push / PR | tests, lint, coverage |
| `ct.yml` | corpus/config/retrieval changes or manual | re-ingest, re-index, evaluate, version artefacts |
| `cd.yml` | `main` changes or manual | build image, smoke-test image, update Kubernetes manifest, optional direct Kind deploy |

## Branching strategy

- `main` — stable production-ready branch.
- `develop` — integration/staging branch.
- `feature/*` — new functionality.
- `experiment/*` — retrieval, chunking, and embedding experiments.
- `hotfix/*` — urgent production fixes.

## Testing

```bash
pytest -v
```

## License

MIT

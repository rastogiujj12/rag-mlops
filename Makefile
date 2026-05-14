.PHONY: install test lint format ingest train evaluate pipeline local-run serve warm-cache docker-build docker-run clean

PDF_DIR ?= data/raw/pdfs
JSONL_PATH ?=
MAX_QUERIES ?= 50

install:
	pip install -r requirements.txt

test:
	pytest -v --cov=src --cov-report=term

lint:
	ruff check src tests app

format:
	ruff check --fix src tests app

# Local PDF corpus workflow: ingest PDFs → build BM25/FAISS indices.
ingest:
	python -m src.cli ingest --source pdf --pdf-dir $(PDF_DIR)

train:
	python -m src.cli train --force

# Requires data/evaluation/eval_questions.json.
evaluate:
	python -m src.cli evaluate --max-queries $(MAX_QUERIES)

# Full gated workflow. Requires local evaluation questions.
pipeline:
	python -m src.cli run-full --source pdf --pdf-dir $(PDF_DIR) --max-queries $(MAX_QUERIES)

# Useful before you have an evaluation set: build artefacts without evaluation.
local-run:
	python -m src.cli run-full --source pdf --pdf-dir $(PDF_DIR) --skip-evaluation

serve:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

warm-cache:
	python scripts/warm_embedding_cache.py

docker-build:
	docker build -t rag-mlops:latest .

docker-run:
	docker run --rm -p 8000:8000 \
		-v $$(pwd)/models/sentence-transformers:/models/sentence-transformers \
		-v $$(pwd)/outputs:/app/outputs \
		-v $$(pwd)/artifacts:/app/artifacts \
		rag-mlops:latest

clean:
	rm -rf data/processed outputs/indices outputs/eval/*.json outputs/eval/*.md outputs/best_model.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

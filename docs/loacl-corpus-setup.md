# Local Corpus Setup

This project uses local PDF files as the input corpus.

## Input corpus

PDF files should be placed in:

```text
data/raw/pdfs/
```

## Pipeline commands

```text
make ingest
make train
make serve
```
## Pipeline staged

1. PDF ingestion
2. Text extraction and cleaning
3. Chunking and metadata extraction
4. BM25 lexical index generation
5. E5 embedding generation
6. FAISS vector index construction
7. FastAPI service deployment

## Notes
The embedding model is cached locally under:
```text
models/sentence-transformers/
```
The first run may download the model. Later runs load it from the local cache.

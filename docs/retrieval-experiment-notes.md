# Retrieval Experiment Notes

The system supports three retrieval approaches:

1. BM25 lexical retrieval
2. E5/FAISS dense vector retrieval
3. Hybrid retrieval using Reciprocal Rank Fusion

## BM25

BM25 is useful for exact keyword and terminology matching.

## E5 + FAISS

E5 generates dense embeddings for semantic similarity. FAISS stores and searches the vector index efficiently.

## Hybrid Retrieval

The hybrid retriever combines BM25 and FAISS results using Reciprocal Rank Fusion. This allows the system to benefit from both lexical and semantic retrieval.

## Selected Strategy

The default system uses hybrid retrieval because it is more robust for academic documents where both exact terminology and semantic similarity matter.

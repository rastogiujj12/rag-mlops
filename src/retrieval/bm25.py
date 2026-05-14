"""BM25 retriever — wraps the BM25Index with the Retriever protocol."""
from __future__ import annotations

import numpy as np

from src.indexing.bm25 import BM25Index, tokenize
from src.retrieval.base import RetrievalResult, Retriever


class BM25Retriever(Retriever):
    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        scores = self.index.bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        # If no token in the query exists in the corpus vocabulary, BM25 returns
        # all-zeros — treat that as a true miss.
        if not np.any(scores > 0):
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        # Keep results with positive score (relative ranking is meaningful within them).
        return [
            RetrievalResult(
                chunk=self.index.chunks[i],
                score=float(scores[i]),
                rank=rank,
                source="bm25",
            )
            for rank, i in enumerate(top_idx)
            if scores[i] > 0
        ]

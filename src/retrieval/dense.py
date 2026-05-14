"""Dense retriever — wraps a DenseIndex."""
from __future__ import annotations

from src.indexing.dense import DenseIndex
from src.retrieval.base import RetrievalResult, Retriever


class DenseRetriever(Retriever):
    def __init__(self, index: DenseIndex):
        self.index = index

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        q = self.index.encode_query(query)
        scores, idxs = self.index.index.search(q, top_k)
        scores, idxs = scores[0], idxs[0]
        return [
            RetrievalResult(
                chunk=self.index.chunks[i],
                score=float(scores[rank]),
                rank=rank,
                source="dense",
            )
            for rank, i in enumerate(idxs)
            if i >= 0
        ]

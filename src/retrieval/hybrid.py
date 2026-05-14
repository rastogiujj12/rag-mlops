"""Hybrid retriever — BM25 + best dense model, fused with RRF, reranked by lexical overlap."""
from __future__ import annotations

from src.core.config import settings
from src.retrieval.base import RetrievalResult, Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import lexical_overlap_rerank, reciprocal_rank_fusion


class HybridRetriever(Retriever):
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int | None = None,
        rerank_weight: float = 0.3,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self.rerank_weight = rerank_weight

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        bm25_hits = self.bm25.retrieve(query, settings.top_k_bm25)
        dense_hits = self.dense.retrieve(query, settings.top_k_dense)

        fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=self.rrf_k)
        reranked = lexical_overlap_rerank(query, fused[: max(top_k * 3, top_k)], weight=self.rerank_weight)
        return reranked[:top_k]

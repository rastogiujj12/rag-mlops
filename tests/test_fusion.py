"""Fusion logic tests."""

from src.chunking.chunk import Chunk
from src.retrieval.base import RetrievalResult
from src.retrieval.fusion import lexical_overlap_rerank, reciprocal_rank_fusion


def _r(cid: str, rank: int, score: float = 1.0, text: str = "x") -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=cid, doc_id=cid, title="", text=text, position=0),
        score=score,
        rank=rank,
    )


def test_rrf_prefers_consistent_top_ranks():
    list_a = [_r("a", 0), _r("b", 1), _r("c", 2)]
    list_b = [_r("a", 0), _r("c", 1), _r("b", 2)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused[0].chunk.chunk_id == "a"  # both lists rank a first
    # b and c have symmetric appearances → either could be 2nd, but a clearly wins


def test_rrf_handles_disjoint_lists():
    list_a = [_r("a", 0), _r("b", 1)]
    list_b = [_r("c", 0), _r("d", 1)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    ids = {r.chunk.chunk_id for r in fused}
    assert ids == {"a", "b", "c", "d"}


def test_lexical_rerank_boosts_overlapping_chunks():
    # "transformer" appears in only one chunk
    results = [
        _r("a", 0, score=0.9, text="aqueducts and roman engineering"),
        _r("b", 1, score=0.8, text="transformer attention mechanism in deep learning"),
    ]
    rer = lexical_overlap_rerank("transformer attention", results, weight=0.7)
    assert rer[0].chunk.chunk_id == "b"


def test_lexical_rerank_noop_on_empty_query():
    results = [_r("a", 0)]
    assert lexical_overlap_rerank("", results) == results

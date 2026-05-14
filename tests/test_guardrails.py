"""Guardrail tests."""
from src.chunking.chunk import Chunk
from src.guardrails import generation_guard, query_guard, retrieval_guard
from src.guardrails.manager import post_generation, post_retrieval, pre_retrieval
from src.retrieval.base import RetrievalResult


def _result(cid: str, text: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=cid, doc_id=cid, title="", text=text, position=0),
        score=score,
        rank=0,
    )


def test_query_guard_blocks_empty():
    assert query_guard.check_query("").allowed is False
    assert query_guard.check_query("   ").allowed is False
    assert query_guard.check_query(None).allowed is False


def test_query_guard_blocks_too_short():
    assert query_guard.check_query("hi").allowed is False


def test_query_guard_blocks_too_long():
    assert query_guard.check_query("x" * 10_000).allowed is False


def test_query_guard_allows_normal_query():
    assert query_guard.check_query("Does aspirin reduce risk of heart attack?").allowed


def test_retrieval_guard_filters_low_scores():
    results = [
        _result("a", "context", score=0.9),
        _result("b", "context", score=0.001),
    ]
    filtered = retrieval_guard.filter_results(results, min_score=0.05)
    assert [r.chunk.chunk_id for r in filtered] == ["a"]


def test_retrieval_guard_deduplicates():
    results = [_result("a", "x", 0.9), _result("a", "x", 0.8)]
    filtered = retrieval_guard.filter_results(results, min_score=0.0)
    assert len(filtered) == 1


def test_coverage_ratio():
    contexts = [_result("a", "aspirin reduces heart attack risk")]
    cov = generation_guard.coverage_ratio("aspirin heart attack", contexts)
    assert cov == 1.0


def test_grounding_score_detects_hallucination():
    contexts = [_result("a", "aspirin reduces cardiovascular risk")]
    grounded = "aspirin reduces cardiovascular risk according to the study"
    hallucinated = "ibuprofen cures cancer in mice models"
    assert generation_guard.grounding_score(grounded, contexts) > 0.5
    assert generation_guard.grounding_score(hallucinated, contexts) < 0.3


def test_manager_pre_retrieval_blocks_bad_query():
    decision = pre_retrieval("")
    assert decision.proceed is False
    assert decision.stage == "query_guard"


def test_manager_post_retrieval_flags_low_coverage():
    contexts = [_result("a", "completely unrelated text", 0.9)]
    decision, _ = post_retrieval("aspirin cardiovascular outcomes", contexts)
    assert decision.proceed is False
    assert decision.stage == "coverage_guard"


def test_manager_post_generation_passes_grounded_answer():
    contexts = [_result("a", "the study shows aspirin reduces heart attack risk")]
    decision = post_generation("aspirin reduces heart attack risk", contexts)
    assert decision.proceed is True

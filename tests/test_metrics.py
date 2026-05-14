"""IR metric unit tests with hand-computed expected values."""
import math

import pytest

from src.evaluation.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    # 2 of 3 relevant docs in top-4
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_precision_at_k_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "c"}
    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_binary_relevance():
    # Retrieved order: relevant at position 1 → DCG = 1, IDCG = 1
    assert ndcg_at_k(["a", "b"], {"a": 1}, 2) == pytest.approx(1.0)


def test_ndcg_graded_matches_formula():
    # rel=2 at pos 2; DCG = (2^2-1)/log2(3) = 3/log2(3)
    retrieved = ["x", "a"]
    relevance = {"a": 2}
    expected_dcg = 3 / math.log2(3)
    expected_idcg = 3 / math.log2(2)
    assert ndcg_at_k(retrieved, relevance, 5) == pytest.approx(expected_dcg / expected_idcg)


def test_aggregate_handles_empty():
    assert aggregate([]) == 0.0
    assert aggregate([0.5, 1.0]) == pytest.approx(0.75)

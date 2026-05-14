"""Guardrails manager — single entry point for the runtime (query-time) guard chain."""
from __future__ import annotations

from dataclasses import dataclass

from src.guardrails import generation_guard, query_guard, retrieval_guard
from src.retrieval.base import RetrievalResult


@dataclass
class GuardrailDecision:
    proceed: bool
    stage: str
    reason: str = ""
    coverage: float | None = None
    grounding: float | None = None


def pre_retrieval(query: str) -> GuardrailDecision:
    outcome = query_guard.check_query(query)
    if not outcome.allowed:
        return GuardrailDecision(False, "query_guard", outcome.reason)
    return GuardrailDecision(True, "query_guard")


def post_retrieval(query: str, results: list[RetrievalResult]) -> tuple[GuardrailDecision, list[RetrievalResult]]:
    filtered = retrieval_guard.filter_results(results)
    if not retrieval_guard.has_minimum_quality(filtered):
        return GuardrailDecision(False, "retrieval_guard", "No high-confidence results"), filtered
    coverage = generation_guard.coverage_ratio(query, filtered)
    if not generation_guard.has_sufficient_coverage(query, filtered):
        return (
            GuardrailDecision(False, "coverage_guard", f"Coverage {coverage:.2f} too low", coverage=coverage),
            filtered,
        )
    return GuardrailDecision(True, "post_retrieval", coverage=coverage), filtered


def post_generation(answer: str, contexts: list[RetrievalResult]) -> GuardrailDecision:
    score = generation_guard.grounding_score(answer, contexts)
    if not generation_guard.is_grounded(answer, contexts):
        return GuardrailDecision(False, "hallucination_guard", f"Grounding {score:.2f} too low", grounding=score)
    return GuardrailDecision(True, "post_generation", grounding=score)

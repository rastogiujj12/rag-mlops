"""Generation guards — coverage (pre-gen) and hallucination (post-gen) checks."""
from __future__ import annotations

from src.core.config import settings
from src.indexing.bm25 import tokenize
from src.retrieval.base import RetrievalResult


def coverage_ratio(query: str, contexts: list[RetrievalResult]) -> float:
    """Fraction of query content tokens that appear in at least one retrieved chunk."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    ctx_tokens: set[str] = set()
    for r in contexts:
        ctx_tokens.update(tokenize(r.chunk.text))
    return len(q_tokens & ctx_tokens) / len(q_tokens)


def has_sufficient_coverage(query: str, contexts: list[RetrievalResult]) -> bool:
    return coverage_ratio(query, contexts) >= settings.min_coverage_ratio


def grounding_score(answer: str, contexts: list[RetrievalResult]) -> float:
    """Fraction of answer content tokens supported by the contexts (token-overlap proxy).

    A token-level proxy is intentionally chosen for its determinism and zero
    runtime cost — production systems may swap in NLI-based fact verification.
    """
    a_tokens = set(tokenize(answer)) - _STOPWORDS
    if not a_tokens:
        return 1.0  # nothing to ground
    ctx_tokens: set[str] = set()
    for r in contexts:
        ctx_tokens.update(tokenize(r.chunk.text))
    return len(a_tokens & ctx_tokens) / len(a_tokens)


def is_grounded(answer: str, contexts: list[RetrievalResult]) -> bool:
    return grounding_score(answer, contexts) >= settings.min_overlap_for_grounding


# Tiny, dependency-free stopword set
_STOPWORDS = {
    "the","a","an","of","to","in","is","are","was","were","be","been","being",
    "and","or","but","if","then","than","for","on","at","by","with","as","that",
    "this","it","its","from","into","about","over","under","again","further",
    "you","your","we","our","i","me","my","they","their","he","she","his","her",
    "do","does","did","done","not","no","so","such","also","may","can","could",
    "will","would","should","shall","might","must",
}

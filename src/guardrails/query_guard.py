"""Query guard. First line of defense — runs before retrieval."""
from __future__ import annotations

from dataclasses import dataclass

from src.indexing.bm25 import tokenize


@dataclass
class GuardOutcome:
    allowed: bool
    reason: str = ""


_MIN_LEN = 3
_MAX_LEN = 512


def check_query(query: str) -> GuardOutcome:
    if query is None:
        return GuardOutcome(False, "Query is null")
    q = query.strip()
    if not q:
        return GuardOutcome(False, "Query is empty")
    if len(q) > _MAX_LEN:
        return GuardOutcome(False, f"Query exceeds {_MAX_LEN} characters")
    tokens = tokenize(q)
    if len(tokens) < _MIN_LEN and len(q) < 15:
        return GuardOutcome(False, "Query too short to retrieve meaningfully")
    return GuardOutcome(True)

"""Deterministic citation/reference helpers.

The LLM is allowed to use source markers like [1], but the backend owns the
source metadata and Harvard-style reference formatting. This prevents invented
bibliographies while keeping answers readable.
"""
from __future__ import annotations

import re
from datetime import date

from src.retrieval.base import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")


def page_label(start_page: int | None, end_page: int | None) -> str | None:
    if start_page and end_page:
        return f"p. {start_page}" if start_page == end_page else f"pp. {start_page}-{end_page}"
    if start_page:
        return f"p. {start_page}"
    return None


def assign_source_ids(results: list[RetrievalResult]) -> list[dict]:
    sources = []
    for idx, result in enumerate(results, start=1):
        chunk = result.chunk
        sources.append(
            {
                "id": idx,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "authors": chunk.authors,
                "year": chunk.year,
                "source_url": chunk.source_url,
                "filename": chunk.filename,
                "pages": page_label(chunk.start_page, chunk.end_page),
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "score": result.score,
                "text": chunk.text[:500],
            }
        )
    return sources


def cited_source_ids(answer: str) -> set[int]:
    return {int(match.group(1)) for match in _CITATION_RE.finditer(answer or "")}


def validate_citations(answer: str, source_count: int, *, require_at_least_one: bool = False) -> bool:
    cited = cited_source_ids(answer)
    if require_at_least_one and not cited:
        return False
    return all(1 <= cid <= source_count for cid in cited)


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{', '.join(authors[:-1])} and {authors[-1]}"


def format_harvard_ul_reference(source: dict, accessed: date | None = None) -> str:
    accessed = accessed or date.today()
    authors = _format_authors(source.get("authors") or [])
    year = source.get("year") or "n.d."
    title = source.get("title") or source.get("doc_id") or "Untitled source"
    url = source.get("source_url")
    pages = source.get("pages")
    page_suffix = f", {pages}" if pages else ""
    if url:
        return f"[{source['id']}] {authors} ({year}) {title}{page_suffix}. Available at: {url} (Accessed: {accessed.strftime('%-d %B %Y')})."
    return f"[{source['id']}] {authors} ({year}) {title}{page_suffix}."


def build_references(sources: list[dict]) -> list[str]:
    return [format_harvard_ul_reference(source) for source in sources]

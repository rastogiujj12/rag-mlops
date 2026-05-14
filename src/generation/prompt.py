"""Prompt assembly for grounded generation with source markers."""
from __future__ import annotations

from src.generation.citations import page_label
from src.retrieval.base import RetrievalResult

UNSUPPORTED_ANSWER = "I cannot answer this from the provided sources."

SYSTEM_PROMPT = f"""You are a careful research assistant.

Rules:
1. Answer ONLY using the provided sources.
2. Do not use prior knowledge or external information.
3. If the answer is not present in the sources, reply exactly: {UNSUPPORTED_ANSWER}
4. Cite claims using only the source markers provided, for example [1] or [2].
5. Do not invent citations, titles, authors, URLs, or page numbers.
6. Keep the answer concise and academic.
"""


def build_prompt(query: str, contexts: list[RetrievalResult]) -> tuple[str, str]:
    blocks = []
    for i, r in enumerate(contexts, start=1):
        chunk = r.chunk
        pages = page_label(chunk.start_page, chunk.end_page) or "page unknown"
        author_year = ""
        if chunk.authors or chunk.year:
            first_author = chunk.authors[0].split(",")[0] if chunk.authors else "Unknown author"
            author_year = f" | {first_author} et al., {chunk.year or 'n.d.'}"
        blocks.append(
            f"SOURCE [{i}]\n"
            f"Title: {chunk.title}\n"
            f"Document ID: {chunk.doc_id}\n"
            f"Pages: {pages}{author_year}\n"
            f"Text:\n{chunk.text}"
        )
    context_block = "\n\n".join(blocks)
    user = f"Sources:\n{context_block}\n\nQuestion: {query}\n\nAnswer with source citations:"
    return SYSTEM_PROMPT, user

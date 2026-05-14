"""Chunking utilities.

The default splitter is paragraph-aware when page text is available, and falls
back to deterministic whitespace windows for JSONL corpora. Metadata is
preserved because citations, evaluation, and rollback all depend on traceable
chunk provenance.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from src.core.config import settings
from src.ingestion.ingest import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int
    filename: str | None = None
    source: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    source_url: str | None = None
    start_page: int | None = None
    end_page: int | None = None
    start_paragraph: int | None = None
    end_paragraph: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_tokens(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    if overlap >= size:
        raise ValueError("overlap must be < chunk size")
    tokens = text.split()
    if not tokens:
        return []
    step = size - overlap
    pieces = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + size >= len(tokens):
            break
    return pieces


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _count_words(text: str) -> int:
    return len(text.split())


def _normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraph_chunks(paragraphs: list[str], chunk_size_words: int, overlap_paragraphs: int = 1) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    i = 0
    while i < len(paragraphs):
        current_parts: list[str] = []
        current_word_count = 0
        start_idx = i
        while i < len(paragraphs):
            para = paragraphs[i]
            para_words = _count_words(para)
            if current_parts and current_word_count + para_words > chunk_size_words:
                break
            current_parts.append(para)
            current_word_count += para_words
            i += 1
        chunks.append(
            {
                "text": _normalize_chunk_text("\n\n".join(current_parts)),
                "start_paragraph": start_idx,
                "end_paragraph": i - 1,
            }
        )
        if i < len(paragraphs):
            i = max(start_idx + 1, i - overlap_paragraphs)
    return chunks


def _estimate_page_range(document: Document, chunk_text: str) -> tuple[int | None, int | None]:
    pages = document.pages or []
    if not pages:
        return None, None
    probe_len = 80
    start_probe = chunk_text[:probe_len].strip()
    end_probe = chunk_text[-probe_len:].strip()
    start_page = None
    end_page = None
    for page in pages:
        page_text = page.get("text", "")
        if start_page is None and start_probe and start_probe[:40] in page_text:
            start_page = page["page_number"]
        if end_probe and end_probe[-40:] in page_text:
            end_page = page["page_number"]
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = start_page
    return start_page, end_page


def _make_chunk(doc: Document, pos: int, text: str, **extra: Any) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{pos}",
        doc_id=doc.doc_id,
        title=doc.title,
        text=text,
        position=pos,
        filename=doc.filename,
        source=doc.source,
        authors=doc.authors,
        year=doc.year,
        source_url=doc.source_url,
        metadata=doc.metadata,
        **extra,
    )


def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    chunks: list[Chunk] = []
    for doc in docs:
        if not (doc.text or "").strip():
            continue

        if doc.pages:
            paragraphs = _split_paragraphs(doc.text)
            raw_chunks = _paragraph_chunks(paragraphs, chunk_size_words=size, overlap_paragraphs=1)
            for pos, raw in enumerate(raw_chunks):
                start_page, end_page = _estimate_page_range(doc, raw["text"])
                chunks.append(
                    _make_chunk(
                        doc,
                        pos,
                        raw["text"],
                        start_page=start_page,
                        end_page=end_page,
                        start_paragraph=raw["start_paragraph"],
                        end_paragraph=raw["end_paragraph"],
                    )
                )
        else:
            body = (doc.title + ". " + doc.text).strip() if doc.title else doc.text
            for pos, piece in enumerate(_split_tokens(body, size, overlap)):
                chunks.append(_make_chunk(doc, pos, piece))
    return chunks

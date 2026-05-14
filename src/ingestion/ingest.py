"""Ingestion: turns local sources into a uniform list of Document objects.

Supported sources:
  - Local PDF directory
  - Pre-existing JSONL corpus

The PDF path intentionally keeps richer metadata than a toy loader because the
API later uses it for grounded citations and Harvard-style references.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import logger


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""
    filename: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    source_url: str | None = None
    pages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Small but effective PDF cleanup borrowed from the earlier CLI prototype.
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_title(first_page_text: str, fallback: str) -> str:
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    if not lines:
        return fallback
    title_lines: list[str] = []
    for line in lines[:20]:
        lower = line.lower()
        if len(line) < 8:
            continue
        if any(marker in lower for marker in ["@", "university", "department", "school", "abstract", "keywords"]):
            break
        title_lines.append(line)
    title = " ".join(title_lines).strip()
    title = re.split(r"\b(Abstract|ABSTRACT|Keywords|Index Terms)\b", title, maxsplit=1)[0].strip()
    return title or fallback


def strip_front_matter_from_first_page(first_page_text: str) -> str:
    lines = [line.rstrip() for line in first_page_text.splitlines()]
    start_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower().strip()
        if lower in {"abstract", "1 introduction", "introduction"} or lower.startswith("abstract "):
            start_idx = i
            break
    cleaned = "\n".join(lines[start_idx:]).strip()
    return clean_text(cleaned) if cleaned else clean_text(first_page_text)


def _load_pdf_metadata(pdf_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional paper metadata from a metadata.json file inside the PDF directory.

    Supported keys: doc_id, filename, title, authors, year, source_url.

    Accepts either:
      - a list of metadata objects, or
      - a dict keyed by doc_id/filename.
    """
    metadata_path = pdf_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {str(item.get("doc_id") or Path(item.get("filename", "")).stem): item for item in raw}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    return {}


def _load_pdfs(pdf_dir: Path) -> list[Document]:
    from pypdf import PdfReader

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    docs: list[Document] = []
    metadata_by_id = _load_pdf_metadata(pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            pages: list[dict[str, Any]] = []
            for page_no, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                cleaned = clean_text(raw)
                if page_no == 1:
                    cleaned = strip_front_matter_from_first_page(cleaned)
                pages.append({"page_number": page_no, "text": cleaned})

            raw_first_page = reader.pages[0].extract_text() if reader.pages else ""
            inferred_title = extract_title(clean_text(raw_first_page or ""), pdf_path.stem.replace("_", " "))
            metadata = metadata_by_id.get(pdf_path.stem, {}) | metadata_by_id.get(pdf_path.name, {})
            text = "\n\n".join(p["text"] for p in pages if p["text"])
            docs.append(
                Document(
                    doc_id=str(metadata.get("doc_id") or pdf_path.stem),
                    title=str(metadata.get("title") or inferred_title),
                    text=text,
                    source=str(pdf_path),
                    filename=pdf_path.name,
                    authors=list(metadata.get("authors") or []),
                    year=metadata.get("year"),
                    source_url=metadata.get("source_url"),
                    pages=pages,
                    metadata={k: v for k, v in metadata.items() if k not in {"doc_id", "title", "authors", "year", "source_url"}},
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse {pdf_path}: {e}")
    logger.info(f"Loaded {len(docs)} PDFs from {pdf_dir}")
    return docs


def _load_jsonl(jsonl_path: Path) -> list[Document]:
    docs: list[Document] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(obj.get("doc_id") or obj.get("id")),
                    title=obj.get("title", ""),
                    text=obj["text"],
                    source=str(obj.get("source") or jsonl_path),
                    filename=obj.get("filename"),
                    authors=obj.get("authors") or [],
                    year=obj.get("year"),
                    source_url=obj.get("source_url"),
                    pages=obj.get("pages") or [],
                    metadata=obj.get("metadata") or {},
                )
            )
    logger.info(f"Loaded {len(docs)} docs from {jsonl_path}")
    return docs


def ingest(
    source: str = "pdf",
    pdf_dir: Path | None = None,
    jsonl_path: Path | None = None,
) -> tuple[list[Document], dict | None, dict | None]:
    """Top-level ingestion. Returns (docs, queries, qrels).

    Queries/qrels are not loaded from PDFs directly. Local evaluation data is
    handled separately by the evaluation pipeline using data/evaluation/*.json.
    """
    if source == "pdf":
        if pdf_dir is None:
            raise ValueError("pdf_dir is required when source='pdf'")
        return _load_pdfs(pdf_dir), None, None
    if source == "jsonl":
        if jsonl_path is None:
            raise ValueError("jsonl_path is required when source='jsonl'")
        return _load_jsonl(jsonl_path), None, None
    raise ValueError(f"Unknown source: {source}. Supported sources: pdf, jsonl")


def save_corpus(docs: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> list[Document]:
    return _load_jsonl(path)

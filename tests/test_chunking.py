"""Chunking unit tests."""
import pytest

from src.chunking.chunk import chunk_documents
from src.ingestion.ingest import Document


def _doc(text: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, title="t", text=text)


def test_chunking_produces_overlapping_windows():
    text = " ".join(f"tok{i}" for i in range(100))
    chunks = chunk_documents([_doc(text)], chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    # Step = 15 → expect ceil((100 - 20)/15) + 1 ~ 7 chunks
    assert all(len(c.text.split()) <= 22 for c in chunks)  # +2 for title prepending
    # Overlap means consecutive chunks share tokens
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert any(t in b for t in a[-5:])


def test_chunking_preserves_doc_id_and_position():
    chunks = chunk_documents([_doc("a b c d e f g h", "doc42")], chunk_size=3, chunk_overlap=1)
    assert all(c.doc_id == "doc42" for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "doc42::0"


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_documents([_doc("a b c")], chunk_size=2, chunk_overlap=2)


def test_chunking_handles_empty_text():
    chunks = chunk_documents([_doc("")], chunk_size=10, chunk_overlap=2)
    assert chunks == []

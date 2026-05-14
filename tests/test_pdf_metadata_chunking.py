from src.chunking.chunk import chunk_documents
from src.ingestion.ingest import Document, clean_text, extract_title


def test_clean_text_repairs_hyphenated_line_breaks():
    assert clean_text("retriev-\nal system") == "retrieval system"


def test_extract_title_uses_first_page_heading():
    text = "A Study of RAG Systems\nUniversity Name\nAbstract\nBody"
    assert extract_title(text, "fallback") == "A Study of RAG Systems"


def test_chunk_documents_preserves_source_metadata_and_page_range():
    doc = Document(
        doc_id="paper1",
        title="Paper One",
        text="Abstract\n\nThis is paragraph one about retrieval.\n\nThis is paragraph two about embeddings.",
        filename="paper1.pdf",
        authors=["Smith, A."],
        year=2024,
        pages=[
            {"page_number": 1, "text": "Abstract\n\nThis is paragraph one about retrieval."},
            {"page_number": 2, "text": "This is paragraph two about embeddings."},
        ],
    )
    chunks = chunk_documents([doc], chunk_size=20, chunk_overlap=0)
    assert chunks
    assert chunks[0].filename == "paper1.pdf"
    assert chunks[0].authors == ["Smith, A."]
    assert chunks[0].year == 2024
    assert chunks[0].start_page is not None

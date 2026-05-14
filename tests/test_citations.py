from src.chunking.chunk import Chunk
from src.generation.citations import assign_source_ids, build_references, validate_citations
from src.retrieval.base import RetrievalResult


def test_source_ids_and_harvard_reference_are_metadata_driven():
    chunk = Chunk(
        chunk_id="doc1::0",
        doc_id="doc1",
        title="Extracting Training Data from Large Language Models",
        text="Carlini et al. discuss extracting memorised training examples.",
        position=0,
        authors=["Carlini, N.", "Tramèr, F."],
        year=2020,
        source_url="https://example.com/paper",
        start_page=3,
        end_page=4,
    )
    sources = assign_source_ids([RetrievalResult(chunk=chunk, score=0.9, rank=0, source="hybrid")])
    references = build_references(sources)

    assert sources[0]["id"] == 1
    assert sources[0]["pages"] == "pp. 3-4"
    assert "Carlini, N. and Tramèr, F. (2020)" in references[0]
    assert "Available at: https://example.com/paper" in references[0]


def test_citation_validation_rejects_unknown_source_id():
    assert validate_citations("Supported claim [1].", source_count=1)
    assert not validate_citations("Unsupported citation [2].", source_count=1)

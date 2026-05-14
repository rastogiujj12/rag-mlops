"""BM25 retrieval test — sanity check that lexical matches surface."""
from src.chunking.chunk import Chunk
from src.indexing.bm25 import BM25Index
from src.retrieval.bm25 import BM25Retriever


def _chunks():
    # Larger corpus so BM25's IDF stays positive (Okapi's IDF goes negative
    # when a term appears in >= half of documents — a real property of BM25,
    # not an artifact of the test).
    raw = [
        ("c1", "machine learning models predict outcomes from data"),
        ("c2", "transformers use attention to model dependencies"),
        ("c3", "the romans built aqueducts across europe"),
        ("c4", "python is a programming language for machine learning"),
        ("c5", "the great wall of china is visible from space"),
        ("c6", "shakespeare wrote hamlet in the early seventeenth century"),
        ("c7", "the mitochondria is the powerhouse of the cell"),
        ("c8", "jazz music originated in new orleans louisiana"),
        ("c9", "photosynthesis converts sunlight into chemical energy"),
        ("c10", "the eiffel tower was completed in 1889 in paris"),
    ]
    return [Chunk(chunk_id=cid, doc_id=cid, title="", text=t, position=0) for cid, t in raw]


def test_bm25_retrieves_lexically_relevant_chunks():
    idx = BM25Index.build(_chunks())
    retriever = BM25Retriever(idx)
    results = retriever.retrieve("machine learning", top_k=3)
    top_ids = [r.chunk.chunk_id for r in results]
    # The two ML chunks should rank above the unrelated chunks
    assert "c1" in top_ids
    assert "c4" in top_ids
    # Top-2 should be the ML chunks, in either order
    assert set(top_ids[:2]) == {"c1", "c4"}


def test_bm25_returns_empty_for_unknown_terms():
    idx = BM25Index.build(_chunks())
    retriever = BM25Retriever(idx)
    results = retriever.retrieve("xyzzy quux", top_k=3)
    assert results == []

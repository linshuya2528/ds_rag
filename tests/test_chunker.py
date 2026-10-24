from rag.chunker import build_chunks, split_by_chars
from rag.schemas import Page


def test_split_by_chars_keeps_overlap():
    parts = split_by_chars("abcdefghij", chunk_size=6, overlap=2)
    assert parts == ["abcdef", "efghij"]


def test_build_chunks_tracks_page_metadata():
    chunks = build_chunks([Page("abc", "book.pdf", 4, "first\n\nsecond")], 20, 2)
    assert chunks[0].id == "abc-p004-c001"
    assert chunks[0].page_start == 4

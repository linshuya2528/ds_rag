from rag.cleaners import clean_pages, paragraphs
from rag.schemas import Page


def test_clean_pages_removes_repeated_margins():
    pages = [
        Page("doc", "book.pdf", 1, "Course Header\nFirst page body\nPage 1"),
        Page("doc", "book.pdf", 2, "Course Header\nSecond page body\nPage 2"),
    ]
    cleaned = clean_pages(pages)
    assert "Course Header" not in cleaned[0].text
    assert "First page body" in cleaned[0].text


def test_paragraphs_merge_wrapped_lines():
    assert paragraphs("one\ntwo\n\nthree") == ["one two", "three"]

import pytest

from tools.text_splitter import split_pages_to_chunks


def test_split_pages_to_chunks_outputs_stable_page_chunks() -> None:
    pages = [
        {"page": 1, "text": "abcdefghij"},
        {"page": 2, "text": "klmnopqrst"},
    ]

    chunks = split_pages_to_chunks(pages, paper_id=42, chunk_size=6, overlap=2)

    assert chunks == [
        {"paper_id": 42, "chunk_id": "42-p1-c1", "page": 1, "chunk_text": "abcdef"},
        {"paper_id": 42, "chunk_id": "42-p1-c2", "page": 1, "chunk_text": "efghij"},
        {"paper_id": 42, "chunk_id": "42-p1-c3", "page": 1, "chunk_text": "ij"},
        {"paper_id": 42, "chunk_id": "42-p2-c1", "page": 2, "chunk_text": "klmnop"},
        {"paper_id": 42, "chunk_id": "42-p2-c2", "page": 2, "chunk_text": "opqrst"},
        {"paper_id": 42, "chunk_id": "42-p2-c3", "page": 2, "chunk_text": "st"},
    ]


def test_split_pages_to_chunks_skips_empty_text() -> None:
    pages = [
        {"page": 1, "text": ""},
        {"page": 2, "text": "   \n\t   "},
        {"page": 3, "text": "useful text"},
    ]

    chunks = split_pages_to_chunks(pages, paper_id="paper-a", chunk_size=50, overlap=0)

    assert chunks == [
        {
            "paper_id": "paper-a",
            "chunk_id": "paper-a-p3-c1",
            "page": 3,
            "chunk_text": "useful text",
        }
    ]


def test_split_pages_to_chunks_validates_window_settings() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        split_pages_to_chunks([{"page": 1, "text": "abc"}], paper_id=1, chunk_size=5, overlap=5)

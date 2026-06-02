import uuid
from pathlib import Path

import fitz

from tools.pdf_parser import parse_pdf


def test_parse_pdf_returns_pages_and_title() -> None:
    pdf_path = _test_pdf_path()
    _write_sample_pdf(pdf_path)

    result = parse_pdf(pdf_path)

    assert result["file_path"] == str(pdf_path)
    assert result["title"] == "ResearchFlow Sample Title"
    assert len(result["pages"]) == 2
    assert result["pages"][0]["page"] == 1
    assert "First page body text." in result["pages"][0]["text"]
    assert result["pages"][1]["page"] == 2
    assert "Second page body text." in result["pages"][1]["text"]


def test_parse_pdf_skips_cover_page_when_extracting_title() -> None:
    pdf_path = _test_pdf_path()
    _write_pdf_with_cover_page(pdf_path)

    result = parse_pdf(pdf_path)

    assert result["title"] == "Better Paper Title From The Real First Page"
    assert len(result["pages"]) == 2


def test_parse_pdf_prefers_metadata_title() -> None:
    pdf_path = _test_pdf_path()
    _write_pdf_with_metadata_title(pdf_path)

    result = parse_pdf(pdf_path)

    assert result["title"] == "Metadata Paper Title"


def test_parse_pdf_extracts_paper_metadata_fields() -> None:
    pdf_path = _test_pdf_path()
    _write_pdf_with_paper_metadata(pdf_path)

    result = parse_pdf(pdf_path)

    assert result["title"] == "Metadata Rich Paper"
    assert result["authors"] == ["Alice Smith", "Bob Chen"]
    assert result["year"] == 2025
    assert result["abstract"] == "This paper introduces a test parser for metadata extraction."
    assert result["metadata"]["page_count"] == 1
    assert result["metadata"]["parser"] == "pymupdf"
    assert result["metadata"]["pdf_metadata"]["title"] == "Metadata Rich Paper"


def _write_sample_pdf(pdf_path: Path) -> None:
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text((72, 72), "ResearchFlow Sample Title")
    first_page.insert_text((72, 110), "First page body text.")

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second page body text.")

    document.save(pdf_path)
    document.close()


def _write_pdf_with_cover_page(pdf_path: Path) -> None:
    document = fitz.open()

    cover = document.new_page()
    cover.insert_text((72, 72), "Technical Report")
    cover.insert_text((72, 110), "Confidential Cover Page")

    first_paper_page = document.new_page()
    first_paper_page.insert_text((72, 72), "Better Paper Title From The Real First Page")
    first_paper_page.insert_text((72, 110), "Abstract")
    first_paper_page.insert_text((72, 140), "This page contains the actual paper body.")

    document.save(pdf_path)
    document.close()


def _write_pdf_with_metadata_title(pdf_path: Path) -> None:
    document = fitz.open()
    document.set_metadata({"title": "Metadata Paper Title"})
    page = document.new_page()
    page.insert_text((72, 72), "Wrong Visible Header")
    page.insert_text((72, 110), "Body text.")
    document.save(pdf_path)
    document.close()


def _write_pdf_with_paper_metadata(pdf_path: Path) -> None:
    document = fitz.open()
    document.set_metadata(
        {
            "title": "Metadata Rich Paper",
            "author": "Alice Smith; Bob Chen",
        }
    )
    page = document.new_page()
    page.insert_text((72, 72), "Fallback Visible Header")
    page.insert_text((72, 110), "Published 2025")
    page.insert_text((72, 140), "Abstract")
    page.insert_text((72, 170), "This paper introduces a test parser for metadata extraction.")
    page.insert_text((72, 210), "1. Introduction")
    page.insert_text((72, 240), "The body starts here.")
    document.save(pdf_path)
    document.close()


def _test_pdf_path() -> Path:
    test_dir = Path("data/test_pdf_parser")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / f"sample-{uuid.uuid4().hex}.pdf"

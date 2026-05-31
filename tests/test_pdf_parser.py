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


def _write_sample_pdf(pdf_path: Path) -> None:
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text((72, 72), "ResearchFlow Sample Title")
    first_page.insert_text((72, 110), "First page body text.")

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second page body text.")

    document.save(pdf_path)
    document.close()


def _test_pdf_path() -> Path:
    test_dir = Path("data/test_pdf_parser")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / f"sample-{uuid.uuid4().hex}.pdf"

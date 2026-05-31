from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz


def parse_pdf(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    pages: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            pages.append(
                {
                    "page": page_index,
                    "text": page.get_text("text").strip(),
                    "file_path": str(path),
                }
            )

    return {
        "file_path": str(path),
        "title": _extract_title(pages),
        "pages": pages,
    }


def _extract_title(pages: list[dict[str, Any]]) -> str | None:
    if not pages:
        return None

    first_page_text = pages[0]["text"]
    for line in first_page_text.splitlines():
        candidate = line.strip()
        if candidate and len(candidate) <= 120:
            return candidate

    return None

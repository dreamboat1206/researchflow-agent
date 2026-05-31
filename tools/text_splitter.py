from __future__ import annotations

from typing import Any


def split_pages_to_chunks(
    pages: list[dict[str, Any]],
    paper_id: int | str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_number = page.get("page")
        text = _normalize_text(str(page.get("text") or ""))
        if not text:
            continue

        page_chunk_index = 0
        start = 0
        step = chunk_size - overlap

        while start < len(text):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                page_chunk_index += 1
                chunks.append(
                    {
                        "paper_id": paper_id,
                        "chunk_id": f"{paper_id}-p{page_number}-c{page_chunk_index}",
                        "page": page_number,
                        "chunk_text": chunk_text,
                    }
                )
            start += step

    return chunks


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


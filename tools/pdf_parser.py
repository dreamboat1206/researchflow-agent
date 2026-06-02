from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


MAX_TITLE_PAGES = 3
MAX_TITLE_LENGTH = 180
MAX_ABSTRACT_PAGES = 3

_TITLE_STOPWORDS = {
    "abstract",
    "introduction",
    "references",
    "acknowledgements",
    "acknowledgments",
    "contents",
    "table of contents",
}

_FRONT_MATTER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^arxiv[:\s]",
        r"^doi[:\s]",
        r"^isbn[:\s]",
        r"^issn[:\s]",
        r"^page\s+\d+",
        r"^\d+\s*$",
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"^(submitted|accepted|published|revised)\b",
        r"\b(copyright|all rights reserved|license|proceedings|conference|workshop)\b",
        r"\b(supplementary|appendix|cover page|technical report)\b",
        r"https?://",
        r"\S+@\S+",
    )
]


def parse_pdf(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    pages: list[dict[str, Any]] = []
    pdf_metadata: dict[str, Any] = {}
    with fitz.open(path) as document:
        pdf_metadata = dict(document.metadata or {})
        for page_index, page in enumerate(document, start=1):
            pages.append(
                {
                    "page": page_index,
                    "text": page.get_text("text").strip(),
                    "file_path": str(path),
                }
            )

    title = _extract_title(pages, metadata_title=str(pdf_metadata.get("title") or "").strip())
    authors = _extract_authors(pages, title, metadata_author=str(pdf_metadata.get("author") or "").strip())
    abstract = _extract_abstract(pages)
    year = _extract_year(pages, pdf_metadata)

    return {
        "file_path": str(path),
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "metadata": {
            "pdf_metadata": _compact_metadata(pdf_metadata),
            "page_count": len(pages),
            "parser": "pymupdf",
        },
        "pages": pages,
    }


def _extract_title(pages: list[dict[str, Any]], metadata_title: str | None = None) -> str | None:
    metadata_candidate = _clean_title_candidate(metadata_title or "")
    if metadata_candidate and _is_valid_title_candidate(metadata_candidate):
        return metadata_candidate

    if not pages:
        return None

    candidates: list[tuple[int, str]] = []
    for page_offset, page in enumerate(pages[:MAX_TITLE_PAGES]):
        lines = [_clean_title_candidate(line) for line in page["text"].splitlines()]
        lines = [line for line in lines if line]
        for index, line in enumerate(lines):
            _add_title_candidate(candidates, line, page_offset)
            if index + 1 < len(lines):
                merged = _clean_title_candidate(f"{line} {lines[index + 1]}")
                _add_title_candidate(candidates, merged, page_offset)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _add_title_candidate(candidates: list[tuple[int, str]], candidate: str, page_offset: int) -> None:
    if not _is_valid_title_candidate(candidate):
        return
    candidates.append((_score_title_candidate(candidate, page_offset), candidate))


def _clean_title_candidate(candidate: str) -> str:
    return re.sub(r"\s+", " ", candidate).strip(" \t\r\n-:|")


def _is_valid_title_candidate(candidate: str) -> bool:
    if not candidate:
        return False
    if len(candidate) < 8 or len(candidate) > MAX_TITLE_LENGTH:
        return False
    lowered = candidate.lower()
    if lowered in _TITLE_STOPWORDS:
        return False
    if any(pattern.search(candidate) for pattern in _FRONT_MATTER_PATTERNS):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z-]*", candidate)
    if len(words) < 2:
        return False
    if len(candidate.split()) > 24:
        return False
    return True


def _score_title_candidate(candidate: str, page_offset: int) -> int:
    words = candidate.split()
    score = 100 - (page_offset * 10)
    if 4 <= len(words) <= 16:
        score += 30
    if 20 <= len(candidate) <= 140:
        score += 20
    if candidate[-1:] not in ".!?":
        score += 10
    else:
        score -= 25
    if any(char.islower() for char in candidate) and any(char.isupper() for char in candidate):
        score += 10
    if candidate.isupper():
        score -= 20
    if ":" in candidate:
        score += 5
    return score


def _extract_authors(
    pages: list[dict[str, Any]],
    title: str | None,
    metadata_author: str | None = None,
) -> list[str] | None:
    metadata_authors = _split_authors(metadata_author or "")
    if metadata_authors:
        return metadata_authors
    if not pages or not title:
        return None

    lines = [_clean_title_candidate(line) for line in pages[0]["text"].splitlines()]
    lines = [line for line in lines if line]
    try:
        title_index = lines.index(title)
    except ValueError:
        return None

    author_lines: list[str] = []
    for line in lines[title_index + 1 : title_index + 6]:
        lowered = line.lower()
        if lowered in _TITLE_STOPWORDS or lowered.startswith("abstract"):
            break
        if any(pattern.search(line) for pattern in _FRONT_MATTER_PATTERNS):
            continue
        if _looks_like_author_line(line):
            author_lines.append(line)
        elif author_lines:
            break

    if not author_lines:
        return None
    return _split_authors(" ".join(author_lines))


def _extract_abstract(pages: list[dict[str, Any]]) -> str | None:
    text = "\n".join(page["text"] for page in pages[:MAX_ABSTRACT_PAGES])
    if not text.strip():
        return None

    match = re.search(
        r"\bAbstract\b\s*[:\n ]+(.*?)(?=\n\s*(?:Introduction|1\.?\s+Introduction|Keywords?|References)\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    abstract = re.sub(r"\s+", " ", match.group(1)).strip(" \t\r\n-")
    return abstract or None


def _extract_year(pages: list[dict[str, Any]], pdf_metadata: dict[str, Any]) -> int | None:
    for key in ("creationDate", "modDate"):
        year = _year_from_text(str(pdf_metadata.get(key) or ""))
        if year:
            return year

    text = "\n".join(page["text"] for page in pages[:MAX_TITLE_PAGES])
    return _year_from_text(text)


def _year_from_text(text: str) -> int | None:
    for match in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    return None


def _split_authors(value: str) -> list[str] | None:
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return None
    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", value)
    authors = [_clean_author_name(part) for part in parts]
    authors = [author for author in authors if author]
    return authors or None


def _clean_author_name(value: str) -> str:
    value = re.sub(r"\S+@\S+", "", value)
    value = re.sub(r"\b\d+\b", "", value)
    value = value.strip(" \t\r\n,;|")
    if not value or len(value) > 80:
        return ""
    return value


def _looks_like_author_line(line: str) -> bool:
    if len(line) > 160:
        return False
    if any(char.isdigit() for char in line):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z.-]*", line)
    if len(words) < 2:
        return False
    lowered = line.lower()
    if any(token in lowered for token in ("university", "institute", "department", "laboratory")):
        return False
    return True


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metadata.items()
        if value not in (None, "")
    }

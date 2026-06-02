from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from storage.qdrant_store import QdrantTextStore
from storage.sqlite_store import DEFAULT_CONFIG_PATH, get_paper


class RetrievalAgent:
    def __init__(
        self,
        text_store: QdrantTextStore | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.text_store = text_store or QdrantTextStore(config_path=config_path)
        self.config_path = config_path

    def search_papers(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        hits = self.text_store.search_text(normalized_query, top_k=top_k)
        return [self._format_hit(hit) for hit in hits]

    def _format_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        paper_id = hit.get("paper_id")
        paper = _safe_get_paper(paper_id, self.config_path)
        title = paper.get("title") if paper else None
        return {
            "title": title or _fallback_title(paper_id),
            "paper_id": paper_id,
            "chunk_id": hit.get("chunk_id"),
            "page": hit.get("page"),
            "chunk_text": hit.get("text") or hit.get("chunk_text"),
            "score": hit.get("score"),
        }


def search_papers(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return RetrievalAgent().search_papers(query, top_k=top_k)


def _safe_get_paper(paper_id: Any, config_path: str | Path) -> dict[str, Any] | None:
    if paper_id is None:
        return None
    try:
        return get_paper(int(paper_id), config_path=config_path)
    except (OSError, ValueError, sqlite3.Error):
        return None


def _fallback_title(paper_id: Any) -> str | None:
    if paper_id is None:
        return None
    return f"Paper {paper_id}"

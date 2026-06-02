from __future__ import annotations

from pathlib import Path
from typing import Any

from storage.qdrant_store import QdrantFigureTextStore
from storage.sqlite_store import DEFAULT_CONFIG_PATH


class FigureRetrievalAgent:
    def __init__(
        self,
        figure_store: QdrantFigureTextStore | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.figure_store = figure_store or QdrantFigureTextStore(config_path=config_path)

    def search_figures_by_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        hits = self.figure_store.search_figures_text(normalized_query, top_k=top_k)
        return [self._format_hit(hit) for hit in hits]

    def _format_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "figure_id": hit.get("figure_id"),
            "paper_id": hit.get("paper_id"),
            "page": hit.get("page"),
            "image_path": hit.get("image_path"),
            "caption": hit.get("caption"),
            "nearby_text": hit.get("nearby_text"),
            "figure_type": hit.get("figure_type"),
            "score": hit.get("score"),
        }


def search_figures_by_text(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return FigureRetrievalAgent().search_figures_by_text(query, top_k=top_k)

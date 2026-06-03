from __future__ import annotations

from pathlib import Path
from typing import Any

from storage.qdrant_store import QdrantFigureImageStore, QdrantFigureTextStore
from storage.sqlite_store import DEFAULT_CONFIG_PATH


FUSION_WEIGHTS = {
    "caption_text_score": 0.5,
    "image_score": 0.3,
    "nearby_text_score": 0.2,
}


class FigureRetrievalAgent:
    def __init__(
        self,
        figure_store: QdrantFigureTextStore | None = None,
        figure_image_store: QdrantFigureImageStore | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.figure_store = figure_store or QdrantFigureTextStore(config_path=config_path)
        self.figure_image_store = figure_image_store or QdrantFigureImageStore(config_path=config_path)

    def search_figures(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "fusion",
    ) -> list[dict[str, Any]]:
        normalized_mode = mode.strip().lower()
        if normalized_mode in {"caption", "caption_text", "text"}:
            return self.search_figures_by_text(query, top_k=top_k)
        if normalized_mode in {"image", "image_text", "image_embedding"}:
            return self.search_figures_by_image_text(query, top_k=top_k)
        if normalized_mode != "fusion":
            raise ValueError("mode must be one of: fusion, text, image")
        return self.search_figures_fusion(query, top_k=top_k)

    def search_figures_by_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        hits = self.figure_store.search_figures_text(normalized_query, top_k=top_k)
        return [self._format_hit(hit) for hit in hits]

    def search_figures_by_image_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        hits = self.figure_image_store.search_figures_by_image_text(normalized_query, top_k=top_k)
        return [self._format_image_hit(hit) for hit in hits]

    def search_figures_fusion(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_k = max(top_k * 3, top_k)
        text_results = self.search_figures_by_text(normalized_query, top_k=candidate_k)
        image_results = self.search_figures_by_image_text(normalized_query, top_k=candidate_k)
        normalized_text_scores = _normalize_scores(text_results)
        normalized_image_scores = _normalize_scores(image_results)

        merged: dict[str, dict[str, Any]] = {}
        for index, result in enumerate(text_results):
            key = _figure_key(result)
            item = merged.setdefault(key, _base_fusion_result(result))
            _merge_metadata(item, result)
            normalized_score = normalized_text_scores[index]
            if result.get("caption"):
                item["score_breakdown"]["caption_text_score"] = normalized_score
            if result.get("nearby_text"):
                item["score_breakdown"]["nearby_text_score"] = normalized_score

        for index, result in enumerate(image_results):
            key = _figure_key(result)
            item = merged.setdefault(key, _base_fusion_result(result))
            _merge_metadata(item, result)
            item["score_breakdown"]["image_score"] = normalized_image_scores[index]

        fused_results = []
        for item in merged.values():
            breakdown = item["score_breakdown"]
            item["final_score"] = (
                FUSION_WEIGHTS["caption_text_score"] * breakdown["caption_text_score"]
                + FUSION_WEIGHTS["image_score"] * breakdown["image_score"]
                + FUSION_WEIGHTS["nearby_text_score"] * breakdown["nearby_text_score"]
            )
            item["score"] = item["final_score"]
            item["score_breakdown"]["weights"] = dict(FUSION_WEIGHTS)
            fused_results.append(item)

        fused_results.sort(key=lambda item: item["final_score"], reverse=True)
        return fused_results[:top_k]

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

    def _format_image_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "figure_id": hit.get("figure_id"),
            "paper_id": hit.get("paper_id"),
            "page": hit.get("page"),
            "image_path": hit.get("image_path"),
            "caption": hit.get("caption"),
            "figure_type": hit.get("figure_type"),
            "score": hit.get("score"),
        }


def search_figures_by_text(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return FigureRetrievalAgent().search_figures_by_text(query, top_k=top_k)


def search_figures_by_image_text(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return FigureRetrievalAgent().search_figures_by_image_text(query, top_k=top_k)


def search_figures(query: str, top_k: int = 5, mode: str = "fusion") -> list[dict[str, Any]]:
    return FigureRetrievalAgent().search_figures(query, top_k=top_k, mode=mode)


def _normalize_scores(results: list[dict[str, Any]]) -> list[float]:
    scores = [float(result.get("score") or 0.0) for result in results]
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if score > 0 else 0.0 for score in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _figure_key(result: dict[str, Any]) -> str:
    return str(result.get("figure_id") or result.get("image_path") or id(result))


def _base_fusion_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "figure_id": result.get("figure_id"),
        "paper_id": result.get("paper_id"),
        "page": result.get("page"),
        "image_path": result.get("image_path"),
        "caption": result.get("caption"),
        "nearby_text": result.get("nearby_text"),
        "figure_type": result.get("figure_type"),
        "final_score": 0.0,
        "score": 0.0,
        "score_breakdown": {
            "caption_text_score": 0.0,
            "image_score": 0.0,
            "nearby_text_score": 0.0,
        },
    }


def _merge_metadata(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("figure_id", "paper_id", "page", "image_path", "caption", "nearby_text", "figure_type"):
        if target.get(key) in (None, "") and source.get(key) not in (None, ""):
            target[key] = source[key]

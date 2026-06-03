from __future__ import annotations

from fastapi.testclient import TestClient

from api.routes_figure_search import get_figure_retrieval_agent
from api.server import app


class FakeFigureRetrievalAgent:
    def search_figures(self, query: str, top_k: int, mode: str = "text") -> list[dict[str, object]]:
        if mode == "fusion":
            return [
                {
                    "figure_id": "7_fig_2_1",
                    "paper_id": 7,
                    "page": 2,
                    "image_path": "data/figures/7/7_fig_2_1.png",
                    "caption": f"Fusion match for {query}",
                    "nearby_text": "Transformer encoder decoder blocks.",
                    "figure_type": "architecture",
                    "score": 0.86,
                    "final_score": 0.86,
                    "score_breakdown": {
                        "caption_text_score": 1.0,
                        "image_score": 0.2,
                        "nearby_text_score": 1.0,
                        "weights": {
                            "caption_text_score": 0.5,
                            "image_score": 0.3,
                            "nearby_text_score": 0.2,
                        },
                    },
                }
            ][:top_k]
        return self.search_figures_by_text(query, top_k)

    def search_figures_by_text(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": f"Figure for {query}",
                "nearby_text": "Transformer encoder decoder blocks.",
                "figure_type": "architecture",
                "score": 0.89,
            }
        ][:top_k]

    def search_figures_by_image_text(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": f"Image match for {query}",
                "figure_type": "architecture",
                "score": 0.91,
            }
        ][:top_k]


def test_search_figures_route_returns_top_k_figures() -> None:
    app.dependency_overrides[get_figure_retrieval_agent] = lambda: FakeFigureRetrievalAgent()
    client = TestClient(app)

    response = client.post("/search/figures", json={"query": "Transformer architecture", "top_k": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "query": "Transformer architecture",
        "top_k": 1,
        "results": [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Figure for Transformer architecture",
                "nearby_text": "Transformer encoder decoder blocks.",
                "figure_type": "architecture",
                "score": 0.89,
            }
        ],
    }


def test_search_figures_route_returns_fusion_scores() -> None:
    app.dependency_overrides[get_figure_retrieval_agent] = lambda: FakeFigureRetrievalAgent()
    client = TestClient(app)

    response = client.post(
        "/search/figures",
        json={"query": "Transformer architecture", "top_k": 1, "mode": "fusion"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["final_score"] == 0.86
    assert result["score_breakdown"] == {
        "caption_text_score": 1.0,
        "image_score": 0.2,
        "nearby_text_score": 1.0,
        "weights": {
            "caption_text_score": 0.5,
            "image_score": 0.3,
            "nearby_text_score": 0.2,
        },
    }


def test_search_figures_by_image_text_route_returns_top_k_figures() -> None:
    app.dependency_overrides[get_figure_retrieval_agent] = lambda: FakeFigureRetrievalAgent()
    client = TestClient(app)

    response = client.post(
        "/search/figures/image-text",
        json={"query": "Transformer architecture", "top_k": 1},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "query": "Transformer architecture",
        "top_k": 1,
        "results": [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Image match for Transformer architecture",
                "figure_type": "architecture",
                "score": 0.91,
            }
        ],
    }

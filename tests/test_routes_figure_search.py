from __future__ import annotations

from fastapi.testclient import TestClient

from api.routes_figure_search import get_figure_retrieval_agent
from api.server import app


class FakeFigureRetrievalAgent:
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

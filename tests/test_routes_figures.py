from __future__ import annotations

from fastapi.testclient import TestClient

import api.routes_figures as routes_figures
from api.server import app


def test_figures_route_returns_caption_and_nearby_text(monkeypatch) -> None:
    monkeypatch.setattr(
        routes_figures,
        "list_figures",
        lambda paper_id=None, title_query=None: [
            {
                "figure_id": "1_fig_2_1",
                "paper_id": 1,
                "paper_title": "ZoomDet",
                "page": 2,
                "figure_type": "other",
                "caption": "Figure 1: Pipeline.",
                "nearby_text": "Nearby context.",
                "image_path": "data/figures/1/1_fig_2_1.png",
            }
        ],
    )
    client = TestClient(app)

    response = client.get("/figures")

    assert response.status_code == 200
    assert response.json() == [
        {
            "figure_id": "1_fig_2_1",
            "paper_id": 1,
            "paper_title": "ZoomDet",
            "page": 2,
            "figure_type": "other",
            "caption": "Figure 1: Pipeline.",
            "nearby_text": "Nearby context.",
            "image_path": "data/figures/1/1_fig_2_1.png",
        }
    ]


def test_figures_route_passes_filters(monkeypatch) -> None:
    calls = []

    def fake_list_figures(paper_id=None, title_query=None):
        calls.append({"paper_id": paper_id, "title_query": title_query})
        return []

    monkeypatch.setattr(routes_figures, "list_figures", fake_list_figures)
    client = TestClient(app)

    response = client.get("/figures?paper_id=3&title=zoom")

    assert response.status_code == 200
    assert response.json() == []
    assert calls == [{"paper_id": 3, "title_query": "zoom"}]

from __future__ import annotations

from fastapi.testclient import TestClient

import api.routes_figures as routes_figures
from api.server import app


def test_figures_route_returns_caption_and_nearby_text(monkeypatch) -> None:
    monkeypatch.setattr(
        routes_figures,
        "list_figures",
        lambda: [
            {
                "figure_id": "1_fig_2_1",
                "paper_id": 1,
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
            "page": 2,
            "figure_type": "other",
            "caption": "Figure 1: Pipeline.",
            "nearby_text": "Nearby context.",
            "image_path": "data/figures/1/1_fig_2_1.png",
        }
    ]

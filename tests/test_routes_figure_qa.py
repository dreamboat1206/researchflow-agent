from __future__ import annotations

from api.routes_figure_qa import get_figure_qa_invoker
from api.server import app
from fastapi.testclient import TestClient


def fake_figure_qa_invoker(
    question: str,
    figure_id: str | None = None,
    query: str | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    return {
        "final_answer": f"Answer for {question}",
        "paper_id": 7,
        "figure_id": figure_id or "7_fig_2_1",
        "page": 2,
        "selected_figure": {
            "figure_id": figure_id or "7_fig_2_1",
            "paper_id": 7,
            "page": 2,
            "image_path": "data/figures/7/7_fig_2_1.png",
            "caption": f"Figure found by {query or figure_id}",
            "nearby_text": "Architecture details.",
            "figure_type": "architecture",
        },
        "citations": [
            {
                "paper_id": 7,
                "figure_id": figure_id or "7_fig_2_1",
                "page": 2,
                "evidence": "Figure caption.",
            }
        ],
        "evaluations": [{"metric_name": "answer_citation_support", "passed": True}],
    }


def test_qa_figure_route_returns_figure_answer() -> None:
    app.dependency_overrides[get_figure_qa_invoker] = lambda: fake_figure_qa_invoker
    client = TestClient(app)

    response = client.post(
        "/qa/figure",
        json={
            "question": "What does this figure show?",
            "figure_id": "7_fig_2_1",
            "top_k": 1,
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "Answer for What does this figure show?"
    assert result["paper_id"] == 7
    assert result["figure_id"] == "7_fig_2_1"
    assert result["page"] == 2
    assert result["selected_figure"]["caption"] == "Figure found by 7_fig_2_1"
    assert result["citations"][0]["figure_id"] == "7_fig_2_1"

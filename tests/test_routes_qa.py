from __future__ import annotations

from fastapi.testclient import TestClient

from api.routes_qa import get_paper_qa_agent
from api.server import app


class FakePaperQAAgent:
    def answer_question(self, question: str, top_k: int) -> dict[str, object]:
        return {
            "answer": f"Answer for {question} [1]",
            "citations": [
                {"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"},
            ],
        }


def test_qa_paper_route_returns_answer_and_citations() -> None:
    app.dependency_overrides[get_paper_qa_agent] = lambda: FakePaperQAAgent()
    client = TestClient(app)

    response = client.post("/qa/paper", json={"question": "What is ZoomDet?", "top_k": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for What is ZoomDet? [1]",
        "citations": [
            {"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"},
        ],
    }

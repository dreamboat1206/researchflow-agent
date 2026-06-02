from __future__ import annotations

from fastapi.testclient import TestClient

from api.routes_paper import get_retrieval_agent
from api.server import app


class FakeRetrievalAgent:
    def search_papers(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "title": "Attention Is All You Need",
                "paper_id": 7,
                "chunk_id": "7-p1-c1",
                "page": 1,
                "chunk_text": f"Result for {query}",
                "score": 0.91,
            }
        ][:top_k]


def test_search_text_route_returns_top_k_chunks() -> None:
    app.dependency_overrides[get_retrieval_agent] = lambda: FakeRetrievalAgent()
    client = TestClient(app)

    response = client.post("/search/text", json={"query": "attention", "top_k": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "query": "attention",
        "top_k": 1,
        "results": [
            {
                "title": "Attention Is All You Need",
                "paper_id": 7,
                "chunk_id": "7-p1-c1",
                "page": 1,
                "chunk_text": "Result for attention",
                "score": 0.91,
            }
        ],
    }

from __future__ import annotations

from agents.retrieval_agent import RetrievalAgent


class FakeTextStore:
    def search_text(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "paper_id": 123456,
                "chunk_id": "123456-p2-c1",
                "page": 2,
                "text": f"matched text for {query}",
                "score": 0.82,
            }
        ][:top_k]


def test_search_papers_formats_qdrant_hits() -> None:
    agent = RetrievalAgent(text_store=FakeTextStore())

    results = agent.search_papers("attention", top_k=1)

    assert results == [
        {
            "title": "Paper 123456",
            "paper_id": 123456,
            "chunk_id": "123456-p2-c1",
            "page": 2,
            "chunk_text": "matched text for attention",
            "score": 0.82,
        }
    ]


def test_search_papers_skips_empty_query() -> None:
    agent = RetrievalAgent(text_store=FakeTextStore())

    assert agent.search_papers("   ", top_k=5) == []

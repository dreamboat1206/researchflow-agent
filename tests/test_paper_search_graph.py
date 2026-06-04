from __future__ import annotations

from graph.paper_search_graph import build_paper_search_graph, invoke_paper_search
from graph.state import create_initial_state


class FakeRouterAgent:
    def route_query(self, user_input: str) -> dict[str, object]:
        return {"task_type": "paper_qa", "query": user_input.strip(), "need_multimodal": False}


class FakeRetrievalAgent:
    def search_papers(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "title": "Attention Is All You Need",
                "paper_id": 1,
                "chunk_id": "1-p1-c1",
                "page": 1,
                "chunk_text": f"Matched {query}",
                "score": 0.9,
            }
        ][:top_k]


def test_build_paper_search_graph_returns_retrieved_chunks() -> None:
    graph = build_paper_search_graph(router_agent=FakeRouterAgent(), retrieval_agent=FakeRetrievalAgent())
    state = create_initial_state("paper_search", user_query="attention", top_k=1)

    result = graph.invoke(state)

    assert result["retrieved_chunks"][0]["chunk_id"] == "1-p1-c1"
    assert [entry["node"] for entry in result["trace"]] == [
        "router_node",
        "text_retrieval_node",
        "format_results_node",
    ]


def test_invoke_paper_search_uses_graph() -> None:
    result = invoke_paper_search(
        "attention",
        top_k=1,
        router_agent=FakeRouterAgent(),
        retrieval_agent=FakeRetrievalAgent(),
    )

    assert result["workflow"] == "paper_search"
    assert result["retrieved_chunks"][0]["title"] == "Attention Is All You Need"

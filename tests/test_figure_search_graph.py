from __future__ import annotations

from graph.figure_search_graph import (
    build_figure_search_graph,
    figure_retrieval_node,
    format_results_node,
    invoke_figure_search,
    router_node,
)
from graph.state import create_initial_state


class FakeRouterAgent:
    def route_query(self, user_input: str) -> dict[str, object]:
        return {
            "task_type": "figure_search",
            "query": user_input.strip(),
            "need_multimodal": True,
        }


class FakeFigureRetrievalAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search_figures(self, query: str, top_k: int, mode: str = "fusion") -> list[dict[str, object]]:
        self.calls.append({"query": query, "top_k": top_k, "mode": mode})
        return [
            {
                "figure_id": "1_fig_2_1",
                "paper_id": 1,
                "page": 2,
                "image_path": "data/figures/1/1_fig_2_1.png",
                "caption": f"Figure for {query}",
                "nearby_text": "Transformer encoder decoder blocks.",
                "figure_type": "architecture",
                "score": 0.86,
                "final_score": 0.86,
                "score_breakdown": {
                    "caption_text_score": 1.0,
                    "image_score": 0.2,
                    "nearby_text_score": 1.0,
                },
                "extra_field": "should be removed by format_results_node",
            }
        ][:top_k]


def test_router_node_writes_figure_search_state() -> None:
    state = create_initial_state("figure_search", user_query=" 找一下 Transformer 架构图 ")

    result = router_node(state, router_agent=FakeRouterAgent())

    assert result["workflow"] == "figure_search"
    assert result["task_type"] == "figure_search"
    assert result["figure_query"] == "找一下 Transformer 架构图"
    assert result["need_multimodal"] is True
    assert result["figure_search_mode"] == "fusion"
    assert result["trace"][-1]["node"] == "router_node"


def test_figure_retrieval_node_writes_retrieved_figures() -> None:
    agent = FakeFigureRetrievalAgent()
    state = create_initial_state("figure_search", user_query="Transformer architecture", top_k=1)
    state["figure_query"] = "Transformer architecture"
    state["task_type"] = "figure_search"
    state["figure_search_mode"] = "fusion"

    result = figure_retrieval_node(state, figure_retrieval_agent=agent)

    assert agent.calls == [{"query": "Transformer architecture", "top_k": 1, "mode": "fusion"}]
    assert result["retrieved_figures"][0]["figure_id"] == "1_fig_2_1"
    assert result["context_figures"][0]["final_score"] == 0.86


def test_format_results_node_keeps_frontend_fields() -> None:
    state = create_initial_state("figure_search", user_query="Transformer architecture")
    state["retrieved_figures"] = FakeFigureRetrievalAgent().search_figures(
        "Transformer architecture",
        top_k=1,
        mode="fusion",
    )

    result = format_results_node(state)

    figure = result["retrieved_figures"][0]
    assert figure == {
        "figure_id": "1_fig_2_1",
        "paper_id": 1,
        "page": 2,
        "image_path": "data/figures/1/1_fig_2_1.png",
        "caption": "Figure for Transformer architecture",
        "nearby_text": "Transformer encoder decoder blocks.",
        "figure_type": "architecture",
        "score": 0.86,
        "final_score": 0.86,
        "score_breakdown": {
            "caption_text_score": 1.0,
            "image_score": 0.2,
            "nearby_text_score": 1.0,
        },
    }


def test_build_figure_search_graph_invokes_all_nodes() -> None:
    graph = build_figure_search_graph(
        router_agent=FakeRouterAgent(),
        figure_retrieval_agent=FakeFigureRetrievalAgent(),
    )
    state = create_initial_state("figure_search", user_query="Transformer architecture", top_k=1)
    state["figure_query"] = "Transformer architecture"
    state["figure_search_mode"] = "fusion"

    result = graph.invoke(state)

    assert result["retrieved_figures"][0]["figure_id"] == "1_fig_2_1"
    assert [entry["node"] for entry in result["trace"]] == [
        "router_node",
        "figure_retrieval_node",
        "format_results_node",
    ]


def test_invoke_figure_search_returns_retrieved_figures() -> None:
    result = invoke_figure_search(
        "Transformer architecture",
        top_k=1,
        mode="fusion",
        router_agent=FakeRouterAgent(),
        figure_retrieval_agent=FakeFigureRetrievalAgent(),
    )

    assert result["workflow"] == "figure_search"
    assert result["retrieved_figures"][0]["image_path"] == "data/figures/1/1_fig_2_1.png"

from __future__ import annotations

from agents.paper_qa_agent import INSUFFICIENT_CONTEXT_ANSWER
from graph.paper_qa_graph import (
    answer_generation_node,
    build_paper_qa_graph,
    evaluation_node,
    invoke_paper_qa,
    router_node,
    text_retrieval_node,
)
from graph.state import create_initial_state


class FakeRouterAgent:
    def __init__(self, task_type: str = "paper_qa"):
        self.task_type = task_type

    def route_query(self, user_input: str) -> dict[str, object]:
        return {
            "task_type": self.task_type,
            "query": user_input.strip(),
            "need_multimodal": False,
        }


class FakeRetrievalAgent:
    def __init__(self, chunks: list[dict[str, object]] | None = None):
        self.chunks = chunks or [
            {
                "title": "ZoomDet",
                "paper_id": 1,
                "chunk_id": "1-p2-c1",
                "page": 2,
                "chunk_text": "ZoomDet adaptively magnifies object regions.",
                "score": 0.91,
            }
        ]
        self.calls: list[dict[str, object]] = []

    def search_papers(self, query: str, top_k: int) -> list[dict[str, object]]:
        self.calls.append({"query": query, "top_k": top_k})
        return self.chunks[:top_k]


class FakeLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "ZoomDet improves UAV detection by magnifying object regions. [1]"


def test_router_node_writes_research_state() -> None:
    state = create_initial_state("paper_qa", user_query=" What does ZoomDet do? ")

    result = router_node(state, router_agent=FakeRouterAgent())

    assert result["workflow"] == "paper_qa"
    assert result["task_type"] == "paper_qa"
    assert result["question"] == "What does ZoomDet do?"
    assert result["need_multimodal"] is False
    assert result["trace"][-1]["node"] == "router_node"


def test_text_retrieval_node_writes_chunks() -> None:
    retrieval_agent = FakeRetrievalAgent()
    state = create_initial_state("paper_qa", user_query="What does ZoomDet do?", top_k=1)
    state["question"] = "What does ZoomDet do?"
    state["task_type"] = "paper_qa"

    result = text_retrieval_node(state, retrieval_agent=retrieval_agent)

    assert retrieval_agent.calls == [{"query": "What does ZoomDet do?", "top_k": 1}]
    assert result["retrieved_chunks"][0]["chunk_id"] == "1-p2-c1"
    assert result["context_chunks"][0]["title"] == "ZoomDet"


def test_answer_generation_node_writes_final_answer_and_citations() -> None:
    llm_client = FakeLLMClient()
    state = create_initial_state("paper_qa", user_query="What does ZoomDet do?")
    state["question"] = "What does ZoomDet do?"
    state["context_chunks"] = FakeRetrievalAgent().chunks

    result = answer_generation_node(state, llm_client=llm_client)

    assert "ZoomDet improves" in result["final_answer"]
    assert result["citations"] == [
        {"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"}
    ]
    assert "What does ZoomDet do?" in result["prompt"]
    assert "chunk_id=1-p2-c1" in llm_client.prompts[0]


def test_answer_generation_node_refuses_without_context() -> None:
    state = create_initial_state("paper_qa", user_query="What is missing?")

    result = answer_generation_node(state, llm_client=FakeLLMClient())

    assert result["final_answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["citations"] == []


def test_evaluation_node_writes_citation_coverage() -> None:
    state = create_initial_state("paper_qa", user_query="What does ZoomDet do?")
    state["final_answer"] = "Answer [1]"
    state["citations"] = [{"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"}]
    state["retrieved_chunks"] = [{"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"}]

    result = evaluation_node(state)

    assert result["evaluations"][-1]["metric_name"] == "answer_citation_support"
    assert result["evaluations"][-1]["passed"] is True


def test_build_paper_qa_graph_invokes_all_nodes() -> None:
    graph = build_paper_qa_graph(
        router_agent=FakeRouterAgent(),
        retrieval_agent=FakeRetrievalAgent(),
        llm_client=FakeLLMClient(),
    )
    state = create_initial_state("paper_qa", user_query="What does ZoomDet do?", top_k=1)
    state["question"] = "What does ZoomDet do?"

    result = graph.invoke(state)

    assert "ZoomDet improves" in result["final_answer"]
    assert result["citations"] == [
        {"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"}
    ]
    assert [entry["node"] for entry in result["trace"]] == [
        "router_node",
        "text_retrieval_node",
        "answer_generation_node",
        "evaluation_node",
    ]


def test_invoke_paper_qa_returns_final_answer_and_citations() -> None:
    result = invoke_paper_qa(
        "What does ZoomDet do?",
        top_k=1,
        router_agent=FakeRouterAgent(),
        retrieval_agent=FakeRetrievalAgent(),
        llm_client=FakeLLMClient(),
    )

    assert "ZoomDet improves" in result["final_answer"]
    assert result["citations"][0]["chunk_id"] == "1-p2-c1"

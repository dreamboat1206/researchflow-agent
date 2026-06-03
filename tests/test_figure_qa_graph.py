from __future__ import annotations

import uuid
from pathlib import Path

from agents.multimodal_qa_agent import FIGURE_NOT_FOUND_ERROR, MultimodalQAAgent, build_figure_qa_prompt
from graph.figure_qa_graph import (
    answer_generation_node,
    build_figure_qa_graph,
    figure_lookup_node,
    invoke_figure_qa,
    router_node,
)
from graph.state import create_initial_state
from storage.sqlite_store import init_db, insert_figure, insert_paper


class FakeRouterAgent:
    def route_query(self, user_input: str) -> dict[str, object]:
        return {"task_type": "figure_qa", "query": user_input.strip(), "need_multimodal": True}


class FakeFigureRetrievalAgent:
    def search_figures(self, query: str, top_k: int, mode: str = "fusion") -> list[dict[str, object]]:
        return [
            {
                "figure_id": "1_fig_2_1",
                "paper_id": 1,
                "page": 2,
                "image_path": "data/figures/1/1_fig_2_1.png",
                "caption": f"Figure for {query}",
                "nearby_text": "The figure shows encoder and decoder attention blocks.",
                "figure_type": "architecture",
                "score": 0.9,
            }
        ][:top_k]


class FakeLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The figure explains the architecture using attention blocks."


def test_router_node_writes_figure_qa_state() -> None:
    state = create_initial_state("figure_qa", user_query="解释一下这个图")

    result = router_node(state, router_agent=FakeRouterAgent())

    assert result["workflow"] == "figure_qa"
    assert result["task_type"] == "figure_qa"
    assert result["question"] == "解释一下这个图"
    assert result["need_multimodal"] is True


def test_figure_lookup_node_finds_figure_by_id() -> None:
    config_path, figure_id = _write_config_with_figure()
    state = create_initial_state("figure_qa", user_query="What does this figure show?")
    state["figure_id"] = figure_id

    result = figure_lookup_node(state, config_path=config_path)

    assert result["selected_figure"]["figure_id"] == figure_id
    assert result["selected_figure"]["paper_id"] == 1
    assert result["retrieved_figures"][0]["caption"] == "Figure 1: Architecture."


def test_figure_lookup_node_uses_query_search_when_no_figure_id() -> None:
    state = create_initial_state("figure_qa", user_query="What does this figure show?", top_k=1)
    state["figure_query"] = "architecture"

    result = figure_lookup_node(state, figure_retrieval_agent=FakeFigureRetrievalAgent())

    assert result["selected_figure"]["figure_id"] == "1_fig_2_1"
    assert result["retrieved_figures"][0]["caption"] == "Figure for architecture"


def test_answer_generation_node_uses_caption_nearby_and_ocr() -> None:
    llm_client = FakeLLMClient()
    agent = MultimodalQAAgent(llm_client=llm_client)
    state = create_initial_state("figure_qa", user_query="What does this figure show?")
    state["question"] = "What does this figure show?"
    state["selected_figure"] = {
        "figure_id": "1_fig_2_1",
        "paper_id": 1,
        "page": 2,
        "caption": "Figure 1: Architecture.",
        "nearby_text": "Encoder decoder attention blocks.",
        "image_path": "data/figures/1/1_fig_2_1.png",
        "metadata": {"ocr_text": "OCR labels: Encoder Decoder"},
    }

    result = answer_generation_node(state, multimodal_qa_agent=agent)

    assert "Reference: paper_id=1; figure_id=1_fig_2_1; page=2" in result["final_answer"]
    assert result["paper_id"] == 1
    assert result["figure_id"] == "1_fig_2_1"
    assert result["page"] == 2
    assert result["citations"] == [
        {
            "paper_id": 1,
            "figure_id": "1_fig_2_1",
            "page": 2,
            "evidence": "Figure 1: Architecture.",
        }
    ]
    assert "OCR labels: Encoder Decoder" in llm_client.prompts[0]


def test_build_figure_qa_graph_invokes_all_nodes_with_query_search() -> None:
    graph = build_figure_qa_graph(
        router_agent=FakeRouterAgent(),
        figure_retrieval_agent=FakeFigureRetrievalAgent(),
        multimodal_qa_agent=MultimodalQAAgent(llm_client=FakeLLMClient()),
    )
    state = create_initial_state("figure_qa", user_query="What does the architecture show?", top_k=1)
    state["question"] = "What does the architecture show?"
    state["figure_query"] = "architecture"

    result = graph.invoke(state)

    assert "Reference: paper_id=1; figure_id=1_fig_2_1; page=2" in result["final_answer"]
    assert result["selected_figure"]["figure_id"] == "1_fig_2_1"
    assert [entry["node"] for entry in result["trace"]] == [
        "router_node",
        "figure_lookup_node",
        "answer_generation_node",
        "evaluation_node",
        "format_output_node",
    ]
    assert result["evaluations"][-1]["metric_name"] == "answer_citation_support"
    assert result["evaluations"][-1]["passed"] is True


def test_invoke_figure_qa_returns_answer_for_figure_id() -> None:
    config_path, figure_id = _write_config_with_figure()
    result = invoke_figure_qa(
        "What does this figure show?",
        figure_id=figure_id,
        router_agent=FakeRouterAgent(),
        multimodal_qa_agent=MultimodalQAAgent(llm_client=FakeLLMClient()),
        config_path=config_path,
    )

    assert result["figure_id"] == figure_id
    assert result["paper_id"] == 1
    assert "Reference: paper_id=1" in result["final_answer"]


def test_invoke_figure_qa_returns_clear_error_when_missing_figure() -> None:
    config_path = _write_config()
    init_db(config_path)

    result = invoke_figure_qa(
        "What does this figure show?",
        figure_id="missing_fig",
        router_agent=FakeRouterAgent(),
        multimodal_qa_agent=MultimodalQAAgent(llm_client=FakeLLMClient()),
        config_path=config_path,
    )

    assert FIGURE_NOT_FOUND_ERROR in result["error"]
    assert result["final_answer"] == FIGURE_NOT_FOUND_ERROR
    assert result["citations"] == []


def test_build_figure_qa_prompt_includes_required_context() -> None:
    prompt = build_figure_qa_prompt(
        "Explain it",
        {
            "paper_id": 7,
            "figure_id": "7_fig_3_2",
            "page": 3,
            "figure_type": "table",
            "caption": "Table 1: Results.",
            "nearby_text": "The table compares AP.",
            "metadata": {"ocr_text": "AP AP50"},
        },
    )

    assert "paper_id: 7" in prompt
    assert "figure_id: 7_fig_3_2" in prompt
    assert "page: 3" in prompt
    assert "caption: Table 1: Results." in prompt
    assert "nearby_text: The table compares AP." in prompt
    assert "ocr_text: AP AP50" in prompt


def _write_config_with_figure() -> tuple[Path, str]:
    config_path = _write_config()
    init_db(config_path)
    paper_id = insert_paper(title="Figure QA Paper", config_path=config_path)
    figure_id = f"{paper_id}_fig_2_1"
    insert_figure(
        paper_id=paper_id,
        figure_index=1,
        page=2,
        figure_id=figure_id,
        figure_type="architecture",
        caption="Figure 1: Architecture.",
        nearby_text="The figure shows attention blocks.",
        image_path="data/figures/1/1_fig_2_1.png",
        metadata={"ocr_text": "Encoder Decoder"},
        config_path=config_path,
    )
    return config_path, figure_id


def _write_config() -> Path:
    test_dir = Path("data/test_figure_qa_graph") / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: researchflow-test.db\n",
        encoding="utf-8",
    )
    return config_path

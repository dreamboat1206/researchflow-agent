from __future__ import annotations

from typing import get_args, get_type_hints

from graph.state import (
    Citation,
    EvaluationResult,
    FigureSearchMode,
    ResearchState,
    RetrievedChunk,
    RetrievedFigure,
    WorkflowName,
    create_initial_state,
)


def test_research_state_defines_shared_workflow_fields() -> None:
    hints = get_type_hints(ResearchState)

    for field in [
        "workflow",
        "run_id",
        "user_query",
        "answer",
        "top_k",
        "retrieved_chunks",
        "retrieved_figures",
        "citations",
        "evaluations",
        "trace",
        "error",
    ]:
        assert field in hints


def test_research_state_covers_ingest_qa_and_figure_workflows() -> None:
    hints = get_type_hints(ResearchState)

    for field in [
        "pdf_path",
        "parsed_pdf",
        "paper_id",
        "chunks",
        "text_vector_count",
        "figure_text_vector_count",
        "figure_image_vector_count",
        "question",
        "prompt",
        "context_chunks",
        "figure_search_mode",
        "figure_query",
        "selected_figure",
        "context_figures",
    ]:
        assert field in hints


def test_retrieved_records_and_citations_have_expected_keys() -> None:
    chunk_hints = get_type_hints(RetrievedChunk)
    figure_hints = get_type_hints(RetrievedFigure)
    citation_hints = get_type_hints(Citation)
    evaluation_hints = get_type_hints(EvaluationResult)

    assert {"paper_id", "chunk_id", "page", "chunk_text", "score"} <= set(chunk_hints)
    assert {"figure_id", "image_path", "caption", "nearby_text", "score", "final_score"} <= set(
        figure_hints
    )
    assert {"title", "page", "chunk_id", "figure_id", "evidence"} <= set(citation_hints)
    assert {"metric_name", "score", "passed", "details"} <= set(evaluation_hints)


def test_workflow_and_figure_search_modes_are_explicit() -> None:
    assert set(get_args(WorkflowName)) == {
        "paper_ingest",
        "paper_qa",
        "figure_search",
        "figure_qa",
    }
    assert set(get_args(FigureSearchMode)) == {"text", "image", "fusion"}


def test_create_initial_state_returns_langgraph_friendly_defaults() -> None:
    state = create_initial_state("figure_search", user_query="Transformer architecture", top_k=3)

    assert state["workflow"] == "figure_search"
    assert state["user_query"] == "Transformer architecture"
    assert state["top_k"] == 3
    assert state["retrieved_chunks"] == []
    assert state["retrieved_figures"] == []
    assert state["citations"] == []
    assert state["evaluations"] == []
    assert state["messages"] == []
    assert state["trace"] == []

    state["retrieved_figures"].append(
        {
            "figure_id": "1_fig_2_1",
            "paper_id": 1,
            "page": 2,
            "image_path": "data/figures/1/1_fig_2_1.png",
            "caption": "Figure 1: Architecture.",
            "score": 0.9,
            "final_score": 0.86,
            "score_breakdown": {"caption_text_score": 1.0, "image_score": 0.2},
        }
    )

    assert state["retrieved_figures"][0]["figure_id"] == "1_fig_2_1"

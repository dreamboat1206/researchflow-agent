from __future__ import annotations

from typing import Any, Literal, TypedDict



WorkflowName = Literal["paper_ingest", "paper_qa", "figure_search", "figure_qa"]
FigureSearchMode = Literal["text", "image", "fusion"]


class RetrievedChunk(TypedDict, total=False):
    title: str | None
    paper_id: int | str | None
    chunk_id: str
    page: int | None
    chunk_text: str
    score: float | None
    metadata: dict[str, Any] | None


class RetrievedFigure(TypedDict, total=False):
    figure_id: str
    paper_id: int | str | None
    page: int | None
    image_path: str
    caption: str | None
    nearby_text: str | None
    figure_type: str | None
    score: float | None
    final_score: float | None
    score_breakdown: dict[str, Any] | None
    metadata: dict[str, Any] | None


class Citation(TypedDict, total=False):
    title: str | None
    paper_id: int | str | None
    page: int | None
    chunk_id: str | None
    figure_id: str | None
    evidence: str | None


class EvaluationResult(TypedDict, total=False):
    metric_name: str
    score: float
    passed: bool
    details: dict[str, Any] | None


class ResearchState(TypedDict, total=False):
    workflow: WorkflowName
    run_id: str
    user_query: str
    task_type: str
    need_multimodal: bool
    answer: str
    final_answer: str
    error: str | None
    messages: list[dict[str, Any]]
    trace: list[dict[str, Any]]

    # Shared retrieval state.
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    retrieved_figures: list[RetrievedFigure]
    citations: list[Citation]
    evaluations: list[EvaluationResult]

    # Paper ingest state.
    pdf_path: str
    parsed_pdf: dict[str, Any]
    paper_id: int
    paper_title: str | None
    chunks: list[dict[str, Any]]
    text_vector_count: int
    figure_count: int
    figure_text_vector_count: int
    figure_image_vector_count: int

    # Paper QA state.
    question: str
    prompt: str
    context_chunks: list[RetrievedChunk]

    # Figure search / QA state.
    figure_id: str
    page: int
    figure_search_mode: FigureSearchMode
    figure_query: str
    selected_figure: RetrievedFigure
    context_figures: list[RetrievedFigure]


def create_initial_state(
    workflow: WorkflowName,
    user_query: str = "",
    top_k: int = 5,
) -> ResearchState:
    return {
        "workflow": workflow,
        "user_query": user_query,
        "top_k": top_k,
        "messages": [],
        "trace": [],
        "retrieved_chunks": [],
        "retrieved_figures": [],
        "citations": [],
        "evaluations": [],
    }

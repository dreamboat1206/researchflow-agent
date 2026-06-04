from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graph.figure_qa_graph import invoke_figure_qa


router = APIRouter(prefix="/qa", tags=["qa"])


class FigureQARequest(BaseModel):
    question: str = Field(..., min_length=1)
    figure_id: str | None = None
    query: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class FigureQAResponse(BaseModel):
    answer: str
    paper_id: int | None = None
    figure_id: str | None = None
    page: int | None = None
    selected_figure: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    error: str | None = None


FigureQAInvoker = Callable[..., dict[str, Any]]


def get_figure_qa_invoker() -> FigureQAInvoker:
    return invoke_figure_qa


@router.post("/figure", response_model=FigureQAResponse)
def qa_figure(
    request: FigureQARequest,
    figure_qa_invoker: FigureQAInvoker = Depends(get_figure_qa_invoker),
) -> dict[str, Any]:
    result = figure_qa_invoker(
        request.question,
        figure_id=request.figure_id,
        query=request.query,
        top_k=request.top_k,
    )
    return {
        "answer": result.get("final_answer") or result.get("answer") or "",
        "paper_id": result.get("paper_id"),
        "figure_id": result.get("figure_id"),
        "page": result.get("page"),
        "selected_figure": result.get("selected_figure"),
        "citations": result.get("citations") or [],
        "evaluations": result.get("evaluations") or [],
        "error": result.get("error"),
    }

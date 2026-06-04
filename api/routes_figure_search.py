from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Any

from agents.figure_retrieval_agent import FigureRetrievalAgent
from graph.figure_search_graph import invoke_figure_search


router = APIRouter(prefix="/search", tags=["search"])


class FigureSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="text")


class FigureSearchResult(BaseModel):
    figure_id: str | None = None
    paper_id: int | str | None = None
    page: int | None = None
    image_path: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    figure_type: str | None = None
    score: float | None = None
    final_score: float | None = None
    score_breakdown: dict[str, Any] | None = None


class FigureImageSearchResult(BaseModel):
    figure_id: str | None = None
    paper_id: int | str | None = None
    page: int | None = None
    image_path: str | None = None
    caption: str | None = None
    figure_type: str | None = None
    score: float | None = None


class FigureSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[FigureSearchResult]


class FigureImageSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[FigureImageSearchResult]


def get_figure_retrieval_agent() -> FigureRetrievalAgent:
    return FigureRetrievalAgent()


@router.post("/figures", response_model=FigureSearchResponse, response_model_exclude_none=True)
def search_figures(
    request: FigureSearchRequest,
    figure_retrieval_agent: FigureRetrievalAgent = Depends(get_figure_retrieval_agent),
) -> FigureSearchResponse:
    graph_state = invoke_figure_search(
        request.query,
        top_k=request.top_k,
        mode=request.mode,  # type: ignore[arg-type]
        figure_retrieval_agent=figure_retrieval_agent,
    )
    results = graph_state.get("retrieved_figures", [])
    return FigureSearchResponse(query=request.query, top_k=request.top_k, results=results)


@router.post("/figures/image-text", response_model=FigureImageSearchResponse)
def search_figures_by_image_text(
    request: FigureSearchRequest,
    figure_retrieval_agent: FigureRetrievalAgent = Depends(get_figure_retrieval_agent),
) -> FigureImageSearchResponse:
    graph_state = invoke_figure_search(
        request.query,
        top_k=request.top_k,
        mode="image",
        figure_retrieval_agent=figure_retrieval_agent,
    )
    results = graph_state.get("retrieved_figures", [])
    return FigureImageSearchResponse(query=request.query, top_k=request.top_k, results=results)

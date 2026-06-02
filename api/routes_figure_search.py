from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.figure_retrieval_agent import FigureRetrievalAgent


router = APIRouter(prefix="/search", tags=["search"])


class FigureSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class FigureSearchResult(BaseModel):
    figure_id: str | None = None
    paper_id: int | str | None = None
    page: int | None = None
    image_path: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    figure_type: str | None = None
    score: float | None = None


class FigureSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[FigureSearchResult]


def get_figure_retrieval_agent() -> FigureRetrievalAgent:
    return FigureRetrievalAgent()


@router.post("/figures", response_model=FigureSearchResponse)
def search_figures(
    request: FigureSearchRequest,
    figure_retrieval_agent: FigureRetrievalAgent = Depends(get_figure_retrieval_agent),
) -> FigureSearchResponse:
    results = figure_retrieval_agent.search_figures_by_text(request.query, top_k=request.top_k)
    return FigureSearchResponse(query=request.query, top_k=request.top_k, results=results)

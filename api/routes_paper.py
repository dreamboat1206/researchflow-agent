from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.retrieval_agent import RetrievalAgent
from graph.paper_search_graph import invoke_paper_search


router = APIRouter(prefix="/search", tags=["search"])


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class TextSearchResult(BaseModel):
    title: str | None = None
    paper_id: int | str | None = None
    chunk_id: str | None = None
    page: int | None = None
    chunk_text: str | None = None
    score: float | None = None


class TextSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[TextSearchResult]


def get_retrieval_agent() -> RetrievalAgent:
    return RetrievalAgent()


@router.post("/text", response_model=TextSearchResponse)
def search_text(
    request: TextSearchRequest,
    retrieval_agent: RetrievalAgent = Depends(get_retrieval_agent),
) -> TextSearchResponse:
    graph_state = invoke_paper_search(
        request.query,
        top_k=request.top_k,
        retrieval_agent=retrieval_agent,
    )
    return TextSearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=graph_state.get("retrieved_chunks", []),
    )

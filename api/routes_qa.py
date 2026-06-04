from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graph.paper_qa_graph import invoke_paper_qa


router = APIRouter(prefix="/qa", tags=["qa"])


class PaperQARequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    title: str | None = None
    page: int | None = None
    chunk_id: str | None = None


class PaperQAResponse(BaseModel):
    answer: str
    citations: list[Citation]


def get_paper_qa_invoker():
    return invoke_paper_qa


@router.post("/paper", response_model=PaperQAResponse)
def qa_paper(
    request: PaperQARequest,
    paper_qa_invoker=Depends(get_paper_qa_invoker),
) -> dict[str, Any]:
    graph_state = paper_qa_invoker(request.question, top_k=request.top_k)
    return {
        "answer": graph_state.get("final_answer") or graph_state.get("answer") or "",
        "citations": graph_state.get("citations") or [],
    }

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.paper_qa_agent import PaperQAAgent


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


def get_paper_qa_agent() -> PaperQAAgent:
    return PaperQAAgent()


@router.post("/paper", response_model=PaperQAResponse)
def qa_paper(
    request: PaperQARequest,
    qa_agent: PaperQAAgent = Depends(get_paper_qa_agent),
) -> dict[str, Any]:
    return qa_agent.answer_question(request.question, top_k=request.top_k)

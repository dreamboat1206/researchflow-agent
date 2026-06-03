from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.paper_qa_agent import PaperQAAgent
from observability.trace_logger import TraceLogger


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
    logger = TraceLogger()
    trace_id, started_at = logger.start_trace()
    try:
        result = qa_agent.answer_question(request.question, top_k=request.top_k)
        logger.log_graph_result(
            trace_id=trace_id,
            started_at=started_at,
            task_type="paper_qa",
            query=request.question,
            state={
                "final_answer": result.get("answer"),
                "citations": result.get("citations") or [],
                "retrieved_chunks": [
                    {
                        "chunk_id": citation.get("chunk_id"),
                        "page": citation.get("page"),
                    }
                    for citation in result.get("citations", [])
                ],
            },
        )
        return result
    except Exception as error:
        logger.log_exception(
            trace_id=trace_id,
            started_at=started_at,
            task_type="paper_qa",
            query=request.question,
            error=error,
        )
        raise

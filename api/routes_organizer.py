from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from agents.organizer_agent import OrganizerAgent


router = APIRouter(prefix="/organizer", tags=["organizer"])


class OrganizePaperRequest(BaseModel):
    dry_run: bool = True
    mode: Literal["copy", "move"] | None = None


class OrganizePapersRequest(BaseModel):
    dry_run: bool = True
    mode: Literal["copy", "move"] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)


def get_organizer_agent() -> OrganizerAgent:
    return OrganizerAgent()


@router.get("/papers")
def list_organizer_papers(
    limit: int | None = Query(default=50, ge=1, le=200),
    organizer_agent: OrganizerAgent = Depends(get_organizer_agent),
) -> dict[str, Any]:
    papers = organizer_agent.sqlite_store.list_papers(limit=limit, config_path=organizer_agent.config_path)
    return {"papers": papers}


@router.post("/papers/{paper_id}")
def organize_paper(
    paper_id: str,
    request: OrganizePaperRequest,
    organizer_agent: OrganizerAgent = Depends(get_organizer_agent),
) -> dict[str, Any]:
    result = organizer_agent.organize_paper(
        paper_id,
        dry_run=request.dry_run,
        mode=request.mode,
    )
    return {"result": result.to_dict()}


@router.post("/papers")
def organize_papers(
    request: OrganizePapersRequest,
    organizer_agent: OrganizerAgent = Depends(get_organizer_agent),
) -> dict[str, Any]:
    results = organizer_agent.organize_all_papers(
        dry_run=request.dry_run,
        mode=request.mode,
        limit=request.limit,
    )
    return {"results": [result.to_dict() for result in results]}

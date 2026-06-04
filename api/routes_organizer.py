from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from agents.organizer_agent import OrganizerAgent
from graph.organizer_graph import invoke_list_organizer_papers, invoke_organize_paper, invoke_organize_papers
from storage.sqlite_store import list_collections, list_paper_tags


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


@router.get("/tags")
def organizer_tags(
    paper_id: int | None = None,
    tag_type: str | None = None,
) -> dict[str, Any]:
    return {"tags": list_paper_tags(paper_id=paper_id, tag_type=tag_type)}


@router.get("/collections")
def organizer_collections() -> dict[str, Any]:
    return {"collections": list_collections()}


@router.get("/papers")
def list_organizer_papers(
    limit: int | None = Query(default=50, ge=1, le=200),
    organizer_agent: OrganizerAgent = Depends(get_organizer_agent),
) -> dict[str, Any]:
    graph_state = invoke_list_organizer_papers(
        limit=limit,
        organizer_agent=organizer_agent,
    )
    return {"papers": graph_state.get("papers", [])}


@router.post("/papers/{paper_id}")
def organize_paper(
    paper_id: str,
    request: OrganizePaperRequest,
    organizer_agent: OrganizerAgent = Depends(get_organizer_agent),
) -> dict[str, Any]:
    try:
        graph_state = invoke_organize_paper(
            paper_id,
            dry_run=request.dry_run,
            mode=request.mode,
            organizer_agent=organizer_agent,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"result": graph_state.get("organization_result", {})}


@router.post("/papers")
def organize_papers(
    request: OrganizePapersRequest,
    organizer_agent: OrganizerAgent = Depends(get_organizer_agent),
) -> dict[str, Any]:
    try:
        graph_state = invoke_organize_papers(
            dry_run=request.dry_run,
            mode=request.mode,
            limit=request.limit,
            organizer_agent=organizer_agent,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"results": graph_state.get("organization_results", [])}

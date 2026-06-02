from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from storage.sqlite_store import list_figures


router = APIRouter(prefix="/figures", tags=["figures"])


class FigureResponse(BaseModel):
    figure_id: str | None = None
    paper_id: int
    paper_title: str | None = None
    page: int | None = None
    figure_type: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    image_path: str | None = None


@router.get("", response_model=list[FigureResponse])
def get_figures(
    paper_id: int | None = Query(default=None),
    title: str | None = Query(default=None),
) -> list[dict[str, object]]:
    return list_figures(paper_id=paper_id, title_query=title)

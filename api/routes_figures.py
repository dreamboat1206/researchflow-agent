from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from storage.sqlite_store import list_figures


router = APIRouter(prefix="/figures", tags=["figures"])


class FigureResponse(BaseModel):
    figure_id: str | None = None
    paper_id: int
    page: int | None = None
    figure_type: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    image_path: str | None = None


@router.get("", response_model=list[FigureResponse])
def get_figures() -> list[dict[str, object]]:
    return list_figures()

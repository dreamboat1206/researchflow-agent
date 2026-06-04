from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from graph.paper_ingest_graph import invoke_paper_ingest
from storage.sqlite_store import (
    DEFAULT_CONFIG_PATH,
    get_chunks_for_paper,
    get_db_path,
    get_paper,
    list_collections,
    list_paper_tags,
    list_papers,
)


router = APIRouter(tags=["papers"])


class IngestOptions(BaseModel):
    auto_organize: bool = True
    organize_mode: Literal["copy", "move"] = "copy"


@router.get("/papers")
def papers(
    limit: int | None = Query(default=100, ge=1, le=500),
    title: str | None = None,
    paper_type: str | None = None,
    primary_topic: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    rows = list_papers(limit=limit)
    filtered = [
        paper
        for paper in rows
        if _matches_paper(paper, title=title, paper_type=paper_type, primary_topic=primary_topic, year=year)
    ]
    return {"papers": filtered}


@router.get("/papers/{paper_id}")
def paper_detail(paper_id: int) -> dict[str, Any]:
    paper = get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
    return {
        "paper": paper,
        "tags": list_paper_tags(paper_id=paper_id),
        "collections": _collections_for_paper(paper_id),
    }


@router.get("/papers/{paper_id}/chunks")
def paper_chunks(
    paper_id: int,
    limit: int = Query(default=20, ge=1, le=500),
) -> dict[str, Any]:
    if get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
    return {"chunks": get_chunks_for_paper(paper_id, limit=limit)}


@router.post("/papers/ingest")
async def ingest_paper(
    file: UploadFile = File(...),
    auto_organize: bool = True,
    organize_mode: Literal["copy", "move"] = "copy",
) -> dict[str, Any]:
    filename = file.filename or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = DEFAULT_CONFIG_PATH.parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / filename
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        with tempfile.NamedTemporaryFile(prefix=f"{stem}-", suffix=suffix, dir=upload_dir, delete=False) as tmp:
            target_path = Path(tmp.name)
            tmp.write(await file.read())
    else:
        target_path.write_bytes(await file.read())

    graph_state = invoke_paper_ingest(
        str(target_path),
        organize=auto_organize,
        organize_mode=organize_mode,
    )
    parsed_pdf = graph_state.get("parsed_pdf") or {}
    organization = graph_state.get("organization") or {}
    return {
        "paper_id": graph_state.get("paper_id"),
        "title": graph_state.get("paper_title"),
        "pages": (parsed_pdf.get("metadata") or {}).get("page_count"),
        "chunks": len(graph_state.get("chunks") or []),
        "vectors": graph_state.get("text_vector_count", 0),
        "collection": graph_state.get("collection"),
        "organization": organization,
        "organization_error": graph_state.get("organization_error"),
    }


@router.get("/stats")
def stats() -> dict[str, Any]:
    db_path = get_db_path(DEFAULT_CONFIG_PATH)
    with __import__("sqlite3").connect(db_path) as connection:
        paper_count = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        organized_count = connection.execute(
            "SELECT COUNT(*) FROM papers WHERE organized_path IS NOT NULL AND organized_path != ''"
        ).fetchone()[0]
    papers_rows = list_papers(limit=5)
    return {
        "total_papers": paper_count,
        "total_chunks": chunk_count,
        "organized_papers": organized_count,
        "recent_papers": papers_rows,
        "recent_organized_papers": [paper for paper in papers_rows if paper.get("organized_path")],
        "sqlite_connected": True,
    }


def _matches_paper(
    paper: dict[str, Any],
    title: str | None,
    paper_type: str | None,
    primary_topic: str | None,
    year: int | None,
) -> bool:
    if title and title.lower() not in str(paper.get("title") or "").lower():
        return False
    if paper_type and paper.get("paper_type") != paper_type:
        return False
    if primary_topic and paper.get("primary_topic") != primary_topic:
        return False
    if year is not None and paper.get("year") != year:
        return False
    return True


def _collections_for_paper(paper_id: int) -> list[dict[str, Any]]:
    collections = []
    paper_id_text = str(paper_id)
    for collection in list_collections():
        # Small local join helper for display only; collection membership stays in storage.
        from storage.sqlite_store import list_collection_items

        if any(str(item.get("paper_id")) == paper_id_text for item in list_collection_items(collection["collection_id"])):
            collections.append(collection)
    return collections

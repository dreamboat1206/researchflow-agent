from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from api.routes_organizer import get_organizer_agent
from api.server import app
from fastapi.testclient import TestClient


@dataclass
class FakeOrganizationResult:
    paper_id: str
    title: str = "Attention Is All You Need"
    paper_type: str = "method"
    primary_topic: str = "transformer"
    year: int = 2017
    tags: list[dict[str, Any]] | None = None
    collections: list[dict[str, Any]] | None = None
    original_path: str = "data/inbox/2307.09288.pdf"
    organized_path: str = "data/library/transformer/2017/Attention_Is_All_You_Need.pdf"
    filename_title_source: str = "title"
    dry_run: bool = True
    action: str = "copy"
    reason: str = "test"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = data["tags"] or []
        data["collections"] = data["collections"] or []
        return data


class FakeSqliteStore:
    def list_papers(self, limit: int | None = None, config_path=None) -> list[dict[str, Any]]:
        return [
            {
                "id": 1,
                "title": "Attention Is All You Need",
                "paper_type": "method",
                "primary_topic": "transformer",
                "year": 2017,
                "original_path": "data/inbox/2307.09288.pdf",
                "organized_path": "data/library/transformer/2017/Attention_Is_All_You_Need.pdf",
                "organization_status": "organized",
                "filename_title_source": "title",
            }
        ][:limit]


class FakeOrganizerAgent:
    config_path = "config.yaml"
    sqlite_store = FakeSqliteStore()

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def organize_paper(self, paper_id: str, dry_run: bool = True, mode: str | None = None):
        self.calls.append({"paper_id": paper_id, "dry_run": dry_run, "mode": mode})
        return FakeOrganizationResult(paper_id=paper_id, dry_run=dry_run, action=mode or "copy")

    def organize_all_papers(
        self,
        dry_run: bool = True,
        mode: str | None = None,
        limit: int | None = None,
    ):
        self.calls.append({"dry_run": dry_run, "mode": mode, "limit": limit})
        return [FakeOrganizationResult(paper_id="1", dry_run=dry_run, action=mode or "copy")]


class MissingPaperOrganizerAgent(FakeOrganizerAgent):
    def organize_paper(self, paper_id: str, dry_run: bool = True, mode: str | None = None):
        raise ValueError(f"Paper not found: {paper_id}")


def test_list_organizer_papers_route_returns_papers() -> None:
    app.dependency_overrides[get_organizer_agent] = lambda: FakeOrganizerAgent()
    client = TestClient(app)

    response = client.get("/organizer/papers?limit=1")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["papers"][0]["organized_path"] == (
        "data/library/transformer/2017/Attention_Is_All_You_Need.pdf"
    )


def test_organize_paper_route_returns_result() -> None:
    app.dependency_overrides[get_organizer_agent] = lambda: FakeOrganizerAgent()
    client = TestClient(app)

    response = client.post("/organizer/papers/1", json={"dry_run": False, "mode": "copy"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["paper_id"] == "1"
    assert result["dry_run"] is False
    assert result["organized_path"] == "data/library/transformer/2017/Attention_Is_All_You_Need.pdf"


def test_organize_paper_route_returns_404_for_missing_paper() -> None:
    app.dependency_overrides[get_organizer_agent] = lambda: MissingPaperOrganizerAgent()
    client = TestClient(app)

    response = client.post("/organizer/papers/999", json={"dry_run": False, "mode": "copy"})

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.json()["detail"] == "Paper not found: 999"


def test_organize_papers_route_returns_batch_results() -> None:
    app.dependency_overrides[get_organizer_agent] = lambda: FakeOrganizerAgent()
    client = TestClient(app)

    response = client.post("/organizer/papers", json={"dry_run": True, "mode": "copy", "limit": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["results"][0]["paper_id"] == "1"

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from graph.organizer_graph import invoke_list_organizer_papers, invoke_organize_paper, invoke_organize_papers


@dataclass
class FakeResult:
    paper_id: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "title": "Paper",
                "paper_type": "method",
                "primary_topic": "transformer",
                "year": 2017,
                "tags": [],
                "collections": [],
                "organized_path": "data/library/transformer/2017/Paper.pdf",
            }
        )
        return data


class FakeSqliteStore:
    def list_papers(self, limit=None, config_path=None):
        return [{"id": 1, "title": "Paper"}][:limit]


class FakeOrganizerAgent:
    config_path = "config.yaml"
    sqlite_store = FakeSqliteStore()

    def organize_paper(self, paper_id: str, dry_run: bool = True, mode: str | None = None):
        return FakeResult(paper_id=paper_id)

    def organize_all_papers(self, dry_run: bool = True, mode: str | None = None, limit: int | None = None):
        return [FakeResult(paper_id="1")]


def test_organizer_graph_lists_papers() -> None:
    result = invoke_list_organizer_papers(limit=1, organizer_agent=FakeOrganizerAgent())

    assert result["papers"] == [{"id": 1, "title": "Paper"}]


def test_organizer_graph_organizes_single_and_batch() -> None:
    single = invoke_organize_paper("1", dry_run=True, mode="copy", organizer_agent=FakeOrganizerAgent())
    batch = invoke_organize_papers(dry_run=True, mode="copy", limit=1, organizer_agent=FakeOrganizerAgent())

    assert single["organization_result"]["paper_id"] == "1"
    assert batch["organization_results"][0]["paper_id"] == "1"

from __future__ import annotations

import uuid
from pathlib import Path

import agents.organizer_agent as organizer_module
import main as cli_main
from agents.organizer_agent import OrganizerAgent, PaperOrganizationResult
from storage.sqlite_store import (
    get_paper,
    init_db,
    insert_chunk,
    insert_paper,
    list_collection_items,
    list_paper_tags,
)


def test_organized_filename_prefers_paper_title_over_original_filename() -> None:
    config_path = _write_config()
    init_db(config_path)
    pdf_path = _write_pdf("2307.09288.pdf")
    paper_id = insert_paper(title="Attention Is All You Need", source_path=str(pdf_path), config_path=config_path)
    agent = OrganizerAgent(config_path=config_path)

    result = agent.organize_paper(str(paper_id), dry_run=True)

    assert Path(result.organized_path or "").name == "Attention_Is_All_You_Need.pdf"
    assert result.filename_title_source == "title"


def test_organized_path_does_not_include_paper_type_folder() -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)

    path = agent.build_organized_path(_result(), "paper.pdf")

    assert path.parts[-3:] == ("transformer", "2017", "Attention_Is_All_You_Need.pdf")
    assert "method" not in path.parts


def test_title_fallback_uses_original_filename_stem() -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)

    title, source = agent.select_existing_title_for_filename({"title": "unknown"}, Path("2307.09288.pdf"))

    assert title == "2307.09288"
    assert source == "original_path.stem"


def test_organizer_reuses_safe_filename_function(monkeypatch) -> None:
    calls = []
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)
    result = _result(title="Unsafe:Title")

    def fake_safe_filename(value: str, max_length: int = 120) -> str:
        calls.append({"value": value, "max_length": max_length})
        return "SAFE_TITLE"

    monkeypatch.setattr(organizer_module, "safe_filename", fake_safe_filename)

    path = agent.build_organized_path(result, "paper.pdf")

    assert calls == [{"value": "Unsafe:Title", "max_length": 120}]
    assert path.name == "SAFE_TITLE.pdf"


def test_safe_filename_handles_windows_illegal_characters() -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)
    result = _result(title='A <Bad>: Title? With / Characters')

    path = agent.build_organized_path(result, "paper.pdf")

    assert path.name == "A_Bad_Title_With_Characters.pdf"


def test_existing_target_gets_numeric_suffix() -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)
    result = _result(title="Attention Is All You Need")
    first_path = agent.build_organized_path(result, "paper.pdf")
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text("existing", encoding="utf-8")

    second_path = agent.build_organized_path(result, "paper.pdf")

    assert second_path.name == "Attention_Is_All_You_Need_1.pdf"


def test_dry_run_does_not_copy_file() -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)
    pdf_path = _write_pdf("dry-run.pdf")
    target_path = Path("data/test_organizer_agent/target") / uuid.uuid4().hex / "dry-run.pdf"

    result_path = agent.apply_file_operation(pdf_path, target_path, mode="copy", dry_run=True)

    assert result_path == target_path
    assert not target_path.exists()


def test_copy_mode_copies_file_and_keeps_original() -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)
    pdf_path = _write_pdf("copy-source.pdf")
    target_path = Path("data/test_organizer_agent/copy") / uuid.uuid4().hex / "copy-target.pdf"

    result_path = agent.apply_file_operation(pdf_path, target_path, mode="copy", dry_run=False)

    assert result_path.exists()
    assert pdf_path.exists()


def test_move_mode_renames_file_without_copying(monkeypatch) -> None:
    config_path = _write_config()
    agent = OrganizerAgent(config_path=config_path)
    pdf_path = _write_pdf("move-source.pdf")
    target_path = Path("data/test_organizer_agent/move") / uuid.uuid4().hex / "move-target.pdf"
    calls = []

    def fake_rename(self: Path, target: Path) -> None:
        calls.append((self, target))
        target.write_bytes(self.read_bytes())

    monkeypatch.setattr(Path, "rename", fake_rename)

    result_path = agent.apply_file_operation(pdf_path, target_path, mode="move", dry_run=False)

    assert result_path.exists()
    assert calls == [(pdf_path, target_path)]


def test_classify_survey_rag_and_transformer() -> None:
    agent = OrganizerAgent(config_path=_write_config())

    rag = agent.classify_paper({"title": "A Survey of Retrieval-Augmented Generation"}, [])
    transformer = agent.classify_paper({"title": "Attention Is All You Need"}, [])

    assert rag["paper_type"] == "survey"
    assert rag["primary_topic"] == "rag"
    assert transformer["primary_topic"] == "transformer"


def test_classify_unknown_defaults() -> None:
    agent = OrganizerAgent(config_path=_write_config())

    result = agent.classify_paper({"title": "Plain Notes"}, [])

    assert result["paper_type"] == "uncategorized"
    assert result["primary_topic"] == "unknown"


def test_apply_result_writes_organization_tags_and_collections() -> None:
    config_path = _write_config()
    init_db(config_path)
    paper_id = insert_paper(title="Attention Is All You Need", source_path="paper.pdf", config_path=config_path)
    result = _result(paper_id=str(paper_id), organized_path="data/library/transformer/2017/title.pdf")
    agent = OrganizerAgent(config_path=config_path)

    agent.apply_result(result)
    agent.apply_result(result)

    paper = get_paper(paper_id, config_path=config_path)
    tags = list_paper_tags(paper_id, config_path=config_path)
    items = list_collection_items("type_method", config_path=config_path)

    assert paper is not None
    assert paper["organized_path"] == "data/library/transformer/2017/title.pdf"
    assert paper["filename_title_source"] == "title"
    assert len([tag for tag in tags if tag["tag"] == "method"]) == 1
    assert len(items) == 1


def test_ingest_pdf_with_organize_calls_organizer(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(cli_main, "init_db", lambda config_path: None)
    monkeypatch.setattr(
        cli_main,
        "parse_pdf",
        lambda file_path: {
            "file_path": file_path,
            "title": "Parsed Title",
            "authors": None,
            "year": None,
            "abstract": None,
            "metadata": {},
            "pages": [{"page": 1, "text": "Parsed text.", "file_path": file_path}],
        },
    )
    monkeypatch.setattr(cli_main, "insert_paper", lambda **kwargs: 7)
    monkeypatch.setattr(
        cli_main,
        "split_pages_to_chunks",
        lambda pages, paper_id, chunk_size, overlap: [
            {"chunk_id": "7-p1-c1", "paper_id": paper_id, "page": 1, "chunk_text": "Parsed text."}
        ],
    )
    monkeypatch.setattr(cli_main, "insert_chunk", lambda **kwargs: 1)
    monkeypatch.setattr(cli_main, "TextEmbeddingModel", lambda config_path: object())
    monkeypatch.setattr(cli_main, "QdrantTextStore", lambda embedding_model, config_path: _FakeQdrantStore())

    class FakeOrganizerAgent:
        def __init__(self, config_path: str):
            self.config_path = config_path

        def organize_paper(self, paper_id: str, dry_run: bool, mode: str | None):
            calls.append({"paper_id": paper_id, "dry_run": dry_run, "mode": mode})
            return _result(paper_id=paper_id)

    monkeypatch.setattr(cli_main, "OrganizerAgent", FakeOrganizerAgent)

    result = cli_main.ingest_pdf("paper.pdf", organize=True, organize_mode="copy")

    assert calls == [{"paper_id": "7", "dry_run": False, "mode": "copy"}]
    assert result["organization"]["paper_id"] == "7"


def _result(
    paper_id: str = "1",
    title: str = "Attention Is All You Need",
    organized_path: str | None = None,
) -> PaperOrganizationResult:
    return PaperOrganizationResult(
        paper_id=paper_id,
        title=title,
        paper_type="method",
        primary_topic="transformer",
        year=2017,
        tags=[
            {"tag": "method", "tag_type": "paper_type", "confidence": 0.9, "source": "organizer"},
            {"tag": "transformer", "tag_type": "topic", "confidence": 0.9, "source": "organizer"},
        ],
        collections=[
            {
                "collection_id": "type_method",
                "name": "Method Papers",
                "description": "",
                "rule": "paper_type=method",
                "reason": "paper_type=method",
            }
        ],
        original_path="paper.pdf",
        organized_path=organized_path,
        filename_title_source="title",
        dry_run=False,
        action="copy",
        reason="test",
    )


def _write_config() -> Path:
    test_dir = Path("data/test_organizer_agent") / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: researchflow-test.db\n"
        "organizer:\n"
        "  library_root: library\n"
        "  default_mode: copy\n"
        "  folder_template: \"{primary_topic}/{year}\"\n"
        "  unknown_year: unknown_year\n"
        "  safe_filename_max_length: 120\n",
        encoding="utf-8",
    )
    return config_path


def _write_pdf(name: str) -> Path:
    path = Path("data/test_organizer_agent/files") / f"{uuid.uuid4().hex}-{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n% test pdf\n")
    return path


class _FakeQdrantStore:
    collection_name = "papers_text"

    def create_collection(self) -> None:
        return None

    def upsert_text_chunks(self, chunks) -> int:
        return len(chunks)

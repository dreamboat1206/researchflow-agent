import sqlite3
import uuid
from pathlib import Path

from storage.sqlite_store import (
    get_paper,
    init_db,
    insert_chunk,
    insert_figure,
    insert_figure_record,
    insert_paper,
    list_papers,
    load_config,
)
from models.figure_record import FigureRecord, FigureType


def test_init_db_creates_expected_tables() -> None:
    config_path = _write_config()

    db_path = init_db(config_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"papers", "chunks", "figures", "traces"}.issubset(table_names)


def test_insert_and_query_paper_chunk_figure() -> None:
    config_path = _write_config()
    db_path = init_db(config_path)

    paper_id = insert_paper(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        source_path="data/papers/attention.pdf",
        abstract="Transformer architecture paper.",
        metadata={"venue": "NeurIPS"},
        config_path=config_path,
    )
    chunk_id = insert_chunk(
        paper_id=paper_id,
        chunk_index=0,
        text="The Transformer is based solely on attention mechanisms.",
        page_start=1,
        page_end=2,
        metadata={"section": "Introduction"},
        config_path=config_path,
    )
    figure_id = insert_figure(
        paper_id=paper_id,
        figure_index=1,
        page=3,
        caption="The Transformer model architecture.",
        image_path="data/figures/attention-figure-1.png",
        figure_type="architecture",
        metadata={"kind": "architecture"},
        config_path=config_path,
    )

    paper = get_paper(paper_id, config_path=config_path)
    papers = list_papers(config_path=config_path)

    assert paper is not None
    assert paper["title"] == "Attention Is All You Need"
    assert paper["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper["metadata"] == {"venue": "NeurIPS"}
    assert papers[0]["id"] == paper_id

    with sqlite3.connect(db_path) as connection:
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        figure_count = connection.execute("SELECT COUNT(*) FROM figures").fetchone()[0]

    assert chunk_id > 0
    assert figure_id > 0
    assert chunk_count == 1
    assert figure_count == 1


def test_insert_figure_record_writes_required_metadata() -> None:
    config_path = _write_config()
    db_path = init_db(config_path)
    paper_id = insert_paper(title="ZoomDet", config_path=config_path)
    record = FigureRecord(
        paper_id=paper_id,
        page=4,
        figure_index=2,
        image_path="data/figures/1_fig_4_2.png",
        caption="ZoomDet pipeline.",
        figure_type=FigureType.PIPELINE,
    )

    row_id = insert_figure_record(record, config_path=config_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM figures WHERE id = ?", (row_id,)).fetchone()

    assert row["figure_id"] == f"{paper_id}_fig_4_2"
    assert row["page"] == 4
    assert row["page_number"] == 4
    assert row["caption"] == "ZoomDet pipeline."
    assert row["image_path"] == "data/figures/1_fig_4_2.png"
    assert row["figure_type"] == "pipeline"


def test_load_config_applies_environment_overrides(monkeypatch) -> None:
    config_path = _write_config()
    monkeypatch.setenv("DATABASE_PATH", "data/override.db")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "papers_text")
    monkeypatch.setenv("EMBEDDING_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("API_PORT", "9000")

    config = load_config(config_path)

    assert config["database"]["path"] == "data/override.db"
    assert config["qdrant"]["url"] == "http://qdrant:6333"
    assert config["qdrant"]["collection_name"] == "papers_text"
    assert config["models"]["embedding_local_files_only"] is True
    assert config["api"]["port"] == 9000


def _write_config() -> Path:
    test_dir = Path("data/test_sqlite_store")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_id = uuid.uuid4().hex
    config_path = test_dir / f"config-{test_id}.yaml"
    config_path.write_text(
        "database:\n"
        f"  path: researchflow-test-{test_id}.db\n",
        encoding="utf-8",
    )
    return config_path

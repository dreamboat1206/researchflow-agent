import sqlite3
import uuid
from pathlib import Path

from storage.sqlite_store import (
    get_paper,
    init_db,
    insert_chunk,
    insert_figure,
    insert_paper,
    list_papers,
)


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
        page_number=3,
        caption="The Transformer model architecture.",
        image_path="data/figures/attention-figure-1.png",
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

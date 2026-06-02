from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import fitz

from storage.sqlite_store import init_db, insert_paper
from tools.figure_extractor import extract_figures_from_pdf


def test_extract_figures_saves_images_and_writes_sqlite() -> None:
    config_path = _write_config()
    db_path = init_db(config_path)
    paper_id = insert_paper(title="Paper With Figures", config_path=config_path)
    pdf_path = Path("data/test_figure_extractor") / f"figures-{uuid.uuid4().hex}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdf_with_images(pdf_path)

    figures = extract_figures_from_pdf(
        pdf_path,
        paper_id=paper_id,
        config_path=config_path,
        min_width=50,
        min_height=50,
    )

    assert len(figures) == 1
    figure = figures[0]
    assert figure["figure_id"] == f"{paper_id}_fig_1_1"
    assert figure["paper_id"] == paper_id
    assert figure["page"] == 1
    assert figure["width"] >= 120
    assert figure["height"] >= 90
    image_path = Path(figure["image_path"])
    assert image_path.exists()
    assert image_path.parent == (config_path.parent / "data/figures" / str(paper_id)).resolve()

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM figures").fetchone()

    assert row["figure_id"] == f"{paper_id}_fig_1_1"
    assert row["paper_id"] == paper_id
    assert row["page"] == 1
    assert row["image_path"] == str(image_path)
    assert row["figure_type"] == "other"


def test_extract_figures_can_skip_sqlite_write() -> None:
    config_path = _write_config()
    db_path = init_db(config_path)
    pdf_path = Path("data/test_figure_extractor") / f"figures-{uuid.uuid4().hex}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdf_with_images(pdf_path)

    figures = extract_figures_from_pdf(
        pdf_path,
        paper_id=99,
        config_path=config_path,
        min_width=50,
        min_height=50,
        write_to_sqlite=False,
    )

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM figures").fetchone()[0]

    assert len(figures) == 1
    assert count == 0


def _write_pdf_with_images(pdf_path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=300, height=300)
    page.insert_image(fitz.Rect(30, 30, 150, 120), stream=_make_png(120, 90))
    page.insert_image(fitz.Rect(180, 30, 190, 40), stream=_make_png(10, 10))
    document.save(pdf_path)
    document.close()


def _make_png(width: int, height: int) -> bytes:
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    pixmap.clear_with(0xAAFFCC)
    return pixmap.tobytes("png")


def _write_config() -> Path:
    test_dir = Path("data/test_figure_extractor") / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: researchflow-test.db\n"
        "storage:\n"
        "  data_dir: data\n",
        encoding="utf-8",
    )
    return config_path

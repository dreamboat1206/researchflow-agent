from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from storage.sqlite_store import (
    init_db,
    insert_figure,
    insert_paper,
    update_figure_caption,
)
from tools.caption_matcher import (
    attach_captions_to_figures,
    extract_captions,
    extract_captions_from_pages,
    infer_figure_type,
)


def test_extract_captions_matches_english_and_chinese() -> None:
    text = """
    Figure 1: Overall architecture of the model.
    Some body text.
    Fig. 2. Detection pipeline.
    Table 3: Ablation results.
    图 4：中文图注示例。
    表 5：中文表格说明。
    """

    captions = extract_captions(text)

    assert [item["caption"] for item in captions] == [
        "Figure 1: Overall architecture of the model.",
        "Fig. 2. Detection pipeline.",
        "Table 3: Ablation results.",
        "图 4：中文图注示例。",
        "表 5：中文表格说明。",
    ]
    assert captions[0]["figure_type"] == "architecture"
    assert captions[2]["figure_type"] == "table"
    assert captions[4]["figure_type"] == "table"
    assert "Some body text" in captions[0]["nearby_text"]


def test_attach_captions_to_figures_uses_nearest_vertical_bbox() -> None:
    captions_by_page = {
        2: [
            {
                "caption": "Table 1: Dataset statistics.",
                "nearby_text": "Table nearby text.",
                "figure_type": "table",
                "bbox": (20.0, 40.0, 280.0, 60.0),
            },
            {
                "caption": "Figure 1: ZoomDet pipeline.",
                "nearby_text": "Figure nearby text.",
                "figure_type": "pipeline",
                "bbox": (20.0, 220.0, 280.0, 240.0),
            },
        ]
    }
    figures = [
        {
            "figure_id": "7_fig_2_1",
            "page": 2,
            "bbox": (40.0, 180.0, 260.0, 210.0),
        }
    ]

    enriched = attach_captions_to_figures(figures, captions_by_page)

    assert enriched[0]["caption"] == "Figure 1: ZoomDet pipeline."
    assert enriched[0]["figure_type"] == "pipeline"
    assert enriched[0]["nearby_text"] == "Figure nearby text."


def test_attach_captions_to_figures_by_page_without_bbox() -> None:
    pages = [
        {"page": 2, "text": "Figure 1: ZoomDet pipeline.\nNearby explanation."},
        {"page": 3, "text": "Table 1: Dataset statistics."},
    ]
    figures = [
        {"figure_id": "7_fig_2_1", "page": 2},
        {"figure_id": "7_fig_3_1", "page": 3},
    ]

    enriched = attach_captions_to_figures(figures, extract_captions_from_pages(pages))

    assert enriched[0]["caption"] == "Figure 1: ZoomDet pipeline."
    assert enriched[1]["caption"] == "Table 1: Dataset statistics."
    assert "Nearby explanation" in enriched[0]["nearby_text"]


def test_infer_figure_type_from_caption_keywords() -> None:
    assert infer_figure_type("Table 1: Results") == "table"
    assert infer_figure_type("图 1：模型架构") == "other"
    assert infer_figure_type("Figure 2: Model architecture") == "architecture"
    assert infer_figure_type("Fig. 3. Detection pipeline") == "pipeline"


def test_update_figure_caption_writes_sqlite() -> None:
    config_path = _write_config()
    db_path = init_db(config_path)
    paper_id = insert_paper(title="Caption Paper", config_path=config_path)
    insert_figure(
        paper_id=paper_id,
        figure_index=1,
        page=2,
        image_path="data/figures/caption.png",
        figure_id=f"{paper_id}_fig_2_1",
        config_path=config_path,
    )

    update_figure_caption(
        f"{paper_id}_fig_2_1",
        "Figure 1: Caption text.",
        "Nearby context text.",
        config_path=config_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT caption, nearby_text FROM figures").fetchone()

    assert row == ("Figure 1: Caption text.", "Nearby context text.")


def _write_config() -> Path:
    test_dir = Path("data/test_caption_matcher") / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: researchflow-test.db\n",
        encoding="utf-8",
    )
    return config_path

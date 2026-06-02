from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from models.figure_record import (
    FigureRecord,
    FigureType,
    build_figure_id,
    default_figure_image_path,
)


def test_build_figure_id_uses_expected_rule() -> None:
    assert build_figure_id(paper_id=7, page=3, index=2) == "7_fig_3_2"


def test_figure_record_generates_id_and_validates_type() -> None:
    record = FigureRecord(
        paper_id=7,
        page=3,
        figure_index=2,
        image_path="data/figures/7_fig_3_2.png",
        caption="A model architecture.",
        figure_type="architecture",
    )

    assert record.figure_id == "7_fig_3_2"
    assert record.figure_type == FigureType.ARCHITECTURE
    assert record.to_metadata()["figure_type"] == "architecture"


def test_figure_record_rejects_invalid_type_and_empty_path() -> None:
    with pytest.raises(ValidationError):
        FigureRecord(
            paper_id=7,
            page=3,
            figure_index=2,
            image_path="",
            figure_type="not-a-type",
        )


def test_default_figure_image_path_uses_figures_directory() -> None:
    assert Path(default_figure_image_path(7, 3, 2)) == Path("data/figures/7_fig_3_2.png")

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class FigureType(str, Enum):
    ARCHITECTURE = "architecture"
    PIPELINE = "pipeline"
    CHART = "chart"
    TABLE = "table"
    ABLATION = "ablation"
    RESULT = "result"
    DATASET = "dataset"
    OTHER = "other"


class FigureRecord(BaseModel):
    paper_id: int
    page: int = Field(..., ge=1)
    figure_index: int = Field(..., ge=1)
    image_path: str
    caption: str | None = None
    figure_type: FigureType = FigureType.OTHER
    figure_id: str | None = None

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image_path must not be empty")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.figure_id is None:
            self.figure_id = build_figure_id(self.paper_id, self.page, self.figure_index)

    def to_metadata(self) -> dict[str, str | int | None]:
        return {
            "figure_id": self.figure_id,
            "figure_type": self.figure_type.value,
            "page": self.page,
            "figure_index": self.figure_index,
            "image_path": self.image_path,
            "caption": self.caption,
        }


def build_figure_id(paper_id: int, page: int, index: int) -> str:
    return f"{paper_id}_fig_{page}_{index}"


def default_figure_image_path(paper_id: int, page: int, index: int, suffix: str = ".png") -> str:
    return str(Path("data/figures") / f"{build_figure_id(paper_id, page, index)}{suffix}")

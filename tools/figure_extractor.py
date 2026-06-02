from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from models.figure_record import FigureType, build_figure_id
from storage.sqlite_store import DEFAULT_CONFIG_PATH, insert_figure, load_config


def extract_figures_from_pdf(
    file_path: str | Path,
    paper_id: int,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    min_width: int = 80,
    min_height: int = 80,
    write_to_sqlite: bool = True,
) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")
    if min_width < 0 or min_height < 0:
        raise ValueError("min_width and min_height must be greater than or equal to 0")

    output_dir = _figure_output_dir(config_path, paper_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[dict[str, Any]] = []
    page_counts: dict[int, int] = {}
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                image = document.extract_image(xref)
                width = int(image.get("width") or 0)
                height = int(image.get("height") or 0)
                if width < min_width or height < min_height:
                    continue

                page_counts[page_index] = page_counts.get(page_index, 0) + 1
                figure_index = page_counts[page_index]
                figure_id = build_figure_id(paper_id, page_index, figure_index)
                extension = _image_extension(image.get("ext"))
                image_path = output_dir / f"{figure_id}.{extension}"
                image_path.write_bytes(image["image"])

                figure = {
                    "figure_id": figure_id,
                    "paper_id": paper_id,
                    "page": page_index,
                    "figure_index": figure_index,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                }
                if write_to_sqlite:
                    insert_figure(
                        paper_id=paper_id,
                        figure_index=figure_index,
                        page=page_index,
                        image_path=str(image_path),
                        figure_id=figure_id,
                        figure_type=FigureType.OTHER,
                        metadata={"width": width, "height": height},
                        config_path=config_path,
                    )
                figures.append(figure)

    return figures


def _figure_output_dir(config_path: str | Path, paper_id: int) -> Path:
    config = load_config(config_path)
    data_dir = Path(config.get("storage", {}).get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = Path(config_path).resolve().parent / data_dir
    return data_dir / "figures" / str(paper_id)


def _image_extension(extension: Any) -> str:
    value = str(extension or "png").lower().lstrip(".")
    if value == "jpeg":
        return "jpg"
    return value or "png"

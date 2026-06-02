from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from models.figure_record import FigureType, build_figure_id
from storage.sqlite_store import DEFAULT_CONFIG_PATH, insert_figure, load_config
from tools.caption_matcher import attach_captions_to_figures, extract_caption_blocks_from_page


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
    captions_by_page: dict[int, list[dict[str, Any]]] = {}
    page_counts: dict[int, int] = {}
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            captions_by_page[page_index] = extract_caption_blocks_from_page(page)
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
                stored_image_path = _stored_image_path(config_path, image_path)

                figure = {
                    "figure_id": figure_id,
                    "paper_id": paper_id,
                    "page": page_index,
                    "figure_index": figure_index,
                    "image_path": stored_image_path,
                    "width": width,
                    "height": height,
                    "bbox": _image_bbox(page, xref),
                }
                figures.append(figure)

    figures = attach_captions_to_figures(figures, captions_by_page)
    figures.extend(
        _extract_text_tables(
            file_path=path,
            paper_id=paper_id,
            output_dir=output_dir,
            config_path=config_path,
            captions_by_page=captions_by_page,
            existing_figures=figures,
            page_counts=page_counts,
        )
    )
    if write_to_sqlite:
        for figure in figures:
            insert_figure(
                paper_id=paper_id,
                figure_index=figure["figure_index"],
                page=figure["page"],
                caption=figure.get("caption"),
                nearby_text=figure.get("nearby_text"),
                image_path=figure["image_path"],
                figure_id=figure["figure_id"],
                figure_type=figure.get("figure_type") or FigureType.OTHER,
                metadata={
                    "width": figure["width"],
                    "height": figure["height"],
                    "bbox": figure.get("bbox"),
                    "caption_bbox": figure.get("caption_bbox"),
                },
                config_path=config_path,
            )

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


def _image_bbox(page: fitz.Page, xref: int) -> tuple[float, float, float, float] | None:
    rects = page.get_image_rects(xref)
    if not rects:
        return None
    rect = rects[0]
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def _extract_text_tables(
    file_path: Path,
    paper_id: int,
    output_dir: Path,
    config_path: str | Path,
    captions_by_page: dict[int, list[dict[str, Any]]],
    existing_figures: list[dict[str, Any]],
    page_counts: dict[int, int],
) -> list[dict[str, Any]]:
    existing_caption_keys = {
        (int(figure.get("page") or 0), figure.get("caption"))
        for figure in existing_figures
        if figure.get("caption")
    }
    table_figures: list[dict[str, Any]] = []
    with fitz.open(file_path) as document:
        for page_index, page in enumerate(document, start=1):
            for caption in captions_by_page.get(page_index, []):
                if caption.get("figure_type") != FigureType.TABLE.value:
                    continue
                if (page_index, caption.get("caption")) in existing_caption_keys:
                    continue

                page_counts[page_index] = page_counts.get(page_index, 0) + 1
                figure_index = page_counts[page_index]
                figure_id = build_figure_id(paper_id, page_index, figure_index)
                clip = _table_clip(page, caption.get("bbox"))
                image_path = output_dir / f"{figure_id}.png"
                pixmap = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2), alpha=False)
                pixmap.save(image_path)
                stored_image_path = _stored_image_path(config_path, image_path)
                table_figures.append(
                    {
                        "figure_id": figure_id,
                        "paper_id": paper_id,
                        "page": page_index,
                        "figure_index": figure_index,
                        "image_path": stored_image_path,
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "bbox": (float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)),
                        "caption": caption.get("caption"),
                        "nearby_text": caption.get("nearby_text"),
                        "caption_bbox": caption.get("bbox"),
                        "figure_type": FigureType.TABLE.value,
                    }
                )
    return table_figures


def _stored_image_path(config_path: str | Path, image_path: Path) -> str:
    base_dir = Path(config_path).resolve().parent
    try:
        return image_path.resolve().relative_to(base_dir).as_posix()
    except ValueError:
        return image_path.as_posix()


def _table_clip(page: fitz.Page, caption_bbox: Any) -> fitz.Rect:
    if not caption_bbox:
        return page.rect
    caption_rect = fitz.Rect(caption_bbox)
    return fitz.Rect(
        0,
        max(0, caption_rect.y0 - 20),
        page.rect.width,
        min(page.rect.height, caption_rect.y1 + 220),
    )

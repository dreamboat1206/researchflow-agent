from __future__ import annotations

import re
from typing import Any

import fitz

from models.figure_record import FigureType


CAPTION_PATTERN = re.compile(
    r"^\s*((?:fig(?:ure)?\.?|table|图|表)\s*[\w\d一二三四五六七八九十百\-\.]*\s*[:：.．、]?\s+.+)$",
    re.IGNORECASE,
)


def extract_captions_from_pages(pages: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    captions_by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        page_number = int(page.get("page") or 0)
        text = str(page.get("text") or "")
        captions = extract_captions(text)
        if captions:
            captions_by_page[page_number] = captions
    return captions_by_page


def extract_captions(page_text: str, nearby_window: int = 240) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    cursor = 0
    for line in page_text.splitlines():
        start = page_text.find(line, cursor)
        end = start + len(line)
        cursor = end
        match = CAPTION_PATTERN.match(line)
        if not match:
            continue
        caption = _normalize_text(match.group(1))
        captions.append(
            {
                "caption": caption,
                "nearby_text": _nearby_text(page_text, start, end, nearby_window),
                "figure_type": infer_figure_type(caption).value,
            }
        )
    return captions


def extract_caption_blocks_from_page(page: fitz.Page) -> list[dict[str, Any]]:
    page_text = page.get_text("text")
    captions: list[dict[str, Any]] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = _line_text(line)
            match = CAPTION_PATTERN.match(line_text)
            if not match:
                continue
            caption = _normalize_text(match.group(1))
            bbox = tuple(float(value) for value in line.get("bbox", block.get("bbox")))
            captions.append(
                {
                    "caption": caption,
                    "nearby_text": _nearby_text_from_page(page, bbox, page_text),
                    "figure_type": infer_figure_type(caption).value,
                    "bbox": bbox,
                }
            )
    return captions


def attach_captions_to_figures(
    figures: list[dict[str, Any]],
    captions_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    used_caption_ids: set[tuple[int, int]] = set()
    for figure in figures:
        page = int(figure.get("page") or figure.get("page_number") or 0)
        page_captions = captions_by_page.get(page, [])
        updated = dict(figure)
        caption_index, caption_match = _nearest_caption(figure, page_captions, page, used_caption_ids)
        if caption_match:
            used_caption_ids.add((page, caption_index))
            updated["caption"] = caption_match["caption"]
            updated["nearby_text"] = caption_match["nearby_text"]
            updated["figure_type"] = caption_match["figure_type"]
            updated["caption_bbox"] = caption_match.get("bbox")
        else:
            updated.setdefault("caption", None)
            updated.setdefault("nearby_text", None)
            updated.setdefault("figure_type", FigureType.OTHER.value)
        enriched.append(updated)
    return enriched


def infer_figure_type(caption: str | None) -> FigureType:
    text = (caption or "").lower()
    stripped = text.strip()
    if stripped.startswith("table") or stripped.startswith("表"):
        return FigureType.TABLE
    if "ablation" in text:
        return FigureType.ABLATION
    if any(keyword in text for keyword in ("dataset", "benchmark", "statistics")):
        return FigureType.DATASET
    if any(keyword in text for keyword in ("result", "comparison", "performance")):
        return FigureType.RESULT
    if any(keyword in text for keyword in ("architecture", "framework", "model structure")):
        return FigureType.ARCHITECTURE
    if any(keyword in text for keyword in ("pipeline", "workflow", "procedure")):
        return FigureType.PIPELINE
    if any(keyword in text for keyword in ("chart", "curve", "plot", "graph")):
        return FigureType.CHART
    return FigureType.OTHER


def _nearest_caption(
    figure: dict[str, Any],
    captions: list[dict[str, Any]],
    page: int,
    used_caption_ids: set[tuple[int, int]],
) -> tuple[int, dict[str, Any] | None]:
    available = [
        (index, caption)
        for index, caption in enumerate(captions)
        if (page, index) not in used_caption_ids
    ]
    if not available:
        return -1, None

    figure_bbox = figure.get("bbox")
    if not figure_bbox:
        return available[0]

    figure_center_y = (_bbox_value(figure_bbox, 1) + _bbox_value(figure_bbox, 3)) / 2
    best_index, best_caption = min(
        available,
        key=lambda item: abs(_caption_center_y(item[1]) - figure_center_y),
    )
    return best_index, best_caption


def _caption_center_y(caption: dict[str, Any]) -> float:
    bbox = caption.get("bbox")
    if not bbox:
        return 0.0
    return (_bbox_value(bbox, 1) + _bbox_value(bbox, 3)) / 2


def _bbox_value(bbox: Any, index: int) -> float:
    return float(bbox[index])


def _line_text(line: dict[str, Any]) -> str:
    return "".join(span.get("text", "") for span in line.get("spans", [])).strip()


def _nearby_text_from_page(page: fitz.Page, bbox: tuple[float, float, float, float], page_text: str) -> str:
    rect = fitz.Rect(bbox)
    expanded = fitz.Rect(
        0,
        max(0, rect.y0 - 120),
        page.rect.width,
        min(page.rect.height, rect.y1 + 160),
    )
    nearby = page.get_textbox(expanded)
    return _normalize_text(nearby or page_text)


def _nearby_text(text: str, start: int, end: int, window: int) -> str:
    nearby_start = max(0, start - window)
    nearby_end = min(len(text), end + window)
    return _normalize_text(text[nearby_start:nearby_end])


def _normalize_text(text: str) -> str:
    return " ".join(text.split())

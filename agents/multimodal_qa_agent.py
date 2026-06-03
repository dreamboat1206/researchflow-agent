from __future__ import annotations

from pathlib import Path
from typing import Any

from models.llm_client import LLMClient
from storage.sqlite_store import DEFAULT_CONFIG_PATH


FIGURE_NOT_FOUND_ERROR = "未找到指定图表，无法回答。"


class MultimodalQAAgent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.llm_client = llm_client or LLMClient(config_path=config_path)

    def answer_figure_question(
        self,
        question: str,
        figure: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_question = question.strip()
        if not figure:
            return {
                "answer": FIGURE_NOT_FOUND_ERROR,
                "error": FIGURE_NOT_FOUND_ERROR,
                "paper_id": None,
                "figure_id": None,
                "page": None,
            }

        prompt = build_figure_qa_prompt(normalized_question, figure)
        answer = self.llm_client.generate(prompt)
        return {
            "answer": _ensure_figure_reference(answer, figure),
            "paper_id": figure.get("paper_id"),
            "figure_id": figure.get("figure_id"),
            "page": figure.get("page") or figure.get("page_number"),
            "caption": figure.get("caption"),
            "image_path": figure.get("image_path"),
        }


def build_figure_qa_prompt(question: str, figure: dict[str, Any]) -> str:
    metadata = figure.get("metadata") or {}
    ocr_text = metadata.get("ocr_text") or metadata.get("ocr") or figure.get("ocr_text") or ""
    return (
        "Use only the figure/table context below to answer the question. "
        "If the context is insufficient, say there is not enough evidence. "
        "The answer must mention paper_id, figure_id, and page.\n\n"
        f"Question:\n{question}\n\n"
        "Figure context:\n"
        f"paper_id: {figure.get('paper_id')}\n"
        f"figure_id: {figure.get('figure_id')}\n"
        f"page: {figure.get('page') or figure.get('page_number')}\n"
        f"figure_type: {figure.get('figure_type') or 'unknown'}\n"
        f"caption: {figure.get('caption') or ''}\n"
        f"nearby_text: {figure.get('nearby_text') or ''}\n"
        f"ocr_text: {ocr_text}\n"
    )


def _ensure_figure_reference(answer: str, figure: dict[str, Any]) -> str:
    required = (
        f"paper_id={figure.get('paper_id')}; "
        f"figure_id={figure.get('figure_id')}; "
        f"page={figure.get('page') or figure.get('page_number')}"
    )
    if str(figure.get("figure_id")) in answer and "paper_id" in answer and "page" in answer:
        return answer
    return f"{answer}\n\nReference: {required}"

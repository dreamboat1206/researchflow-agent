from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, TypedDict

from models.llm_client import LLMClient
from storage.sqlite_store import DEFAULT_CONFIG_PATH


TaskType = Literal["paper_qa", "figure_search", "figure_qa", "paper_ingest", "organize", "unknown"]
VALID_TASK_TYPES: set[str] = {
    "paper_qa",
    "figure_search",
    "figure_qa",
    "paper_ingest",
    "organize",
    "unknown",
}


class RouteResult(TypedDict):
    task_type: TaskType
    query: str
    need_multimodal: bool


class RouterAgent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.llm_client = llm_client or LLMClient(config_path=config_path)

    def route_query(self, user_input: str) -> RouteResult:
        query = _normalize_query(user_input)
        if not query:
            return _route_result("unknown", query, need_multimodal=False)

        rule_result = _route_by_rules(query)
        if rule_result["task_type"] != "unknown":
            return rule_result

        fallback_task = self._route_by_llm(query)
        if fallback_task in VALID_TASK_TYPES:
            return _route_result(fallback_task, query, _need_multimodal(fallback_task, query))
        return _route_result("unknown", query, need_multimodal=False)

    def _route_by_llm(self, query: str) -> str:
        prompt = (
            "Classify the user request into exactly one task type:\n"
            "paper_qa, figure_search, figure_qa, paper_ingest, organize, unknown.\n"
            "Return only the task type.\n\n"
            f"User request: {query}"
        )
        response = self.llm_client.generate(prompt).strip().lower()
        match = re.search(
            r"\b(paper_qa|figure_search|figure_qa|paper_ingest|organize|unknown)\b",
            response,
        )
        return match.group(1) if match else "unknown"


def route_query(user_input: str) -> RouteResult:
    return RouterAgent().route_query(user_input)


def _route_by_rules(query: str) -> RouteResult:
    lowered = query.lower()

    if _contains_any(lowered, ["upload", "ingest", "import", "add paper", "parse pdf", "入库", "导入", "上传"]):
        return _route_result("paper_ingest", query, need_multimodal=False)

    if _contains_any(lowered, ["整理", "归档", "organize", "deduplicate", "分类", "标签"]):
        return _route_result("organize", query, need_multimodal=False)

    if _looks_like_figure_qa(lowered):
        return _route_result("figure_qa", query, need_multimodal=True)

    if _looks_like_figure_search(lowered):
        return _route_result("figure_search", query, need_multimodal=True)

    if _looks_like_paper_qa(lowered):
        return _route_result("paper_qa", query, need_multimodal=False)

    return _route_result("unknown", query, need_multimodal=False)


def _route_result(task_type: str, query: str, need_multimodal: bool) -> RouteResult:
    return {
        "task_type": task_type,  # type: ignore[typeddict-item]
        "query": query,
        "need_multimodal": need_multimodal,
    }


def _normalize_query(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input).strip()


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _looks_like_figure_search(text: str) -> bool:
    return _contains_any(
        text,
        [
            "find figure",
            "find table",
            "search figure",
            "search table",
            "architecture diagram",
            "pipeline figure",
            "找图",
            "找一下图",
            "检索图",
            "搜索图",
            "架构图",
            "流程图",
            "图表",
            "表格",
        ],
    )


def _looks_like_figure_qa(text: str) -> bool:
    has_figure_reference = _contains_any(
        text,
        ["figure", "fig.", "table", "图", "表", "这张图", "这个图", "这张表", "这个表"],
    )
    has_question_intent = _contains_any(
        text,
        ["what", "why", "how", "explain", "describe", "meaning", "说明", "解释", "什么意思", "为什么", "如何"],
    )
    return has_figure_reference and has_question_intent


def _looks_like_paper_qa(text: str) -> bool:
    return _contains_any(
        text,
        [
            "what",
            "why",
            "how",
            "summarize",
            "compare",
            "method",
            "contribution",
            "论文",
            "方法",
            "贡献",
            "摘要",
            "总结",
            "对比",
            "回答",
            "问答",
        ],
    )


def _need_multimodal(task_type: str, query: str) -> bool:
    if task_type in {"figure_search", "figure_qa"}:
        return True
    return _looks_like_figure_search(query) or _looks_like_figure_qa(query)

from __future__ import annotations

from typing import Any

from graph.state import EvaluationResult, ResearchState


REFUSAL_MARKERS = [
    "未找到",
    "无法回答",
    "没有足够",
    "不足",
    "not enough",
    "insufficient",
    "do not have enough",
    "cannot answer",
]


class EvaluatorAgent:
    def evaluate_state(self, state: ResearchState) -> EvaluationResult:
        answer = str(state.get("final_answer") or state.get("answer") or "")
        citations = list(state.get("citations") or [])
        retrieved_chunks = list(state.get("retrieved_chunks") or state.get("context_chunks") or [])
        retrieved_figures = list(state.get("retrieved_figures") or state.get("context_figures") or [])
        return self.evaluate_answer(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            retrieved_figures=retrieved_figures,
        )

    def evaluate_answer(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]] | None = None,
        retrieved_figures: list[dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        retrieved_chunks = retrieved_chunks or []
        retrieved_figures = retrieved_figures or []
        has_retrieval = bool(retrieved_chunks or retrieved_figures)

        if not has_retrieval:
            refused = _is_refusal(answer)
            return _evaluation_result(
                passed=refused,
                reason="no_retrieval_refused" if refused else "no_retrieval_but_answered",
                citation_count=len(citations),
                retrieved_chunk_count=0,
                retrieved_figure_count=0,
            )

        if not citations:
            return _evaluation_result(
                passed=False,
                reason="missing_citations",
                citation_count=0,
                retrieved_chunk_count=len(retrieved_chunks),
                retrieved_figure_count=len(retrieved_figures),
            )

        unsupported = [
            citation
            for citation in citations
            if not _citation_supported(citation, retrieved_chunks, retrieved_figures)
        ]
        passed = not unsupported
        return _evaluation_result(
            passed=passed,
            reason="citations_supported" if passed else "unsupported_citations",
            citation_count=len(citations),
            retrieved_chunk_count=len(retrieved_chunks),
            retrieved_figure_count=len(retrieved_figures),
            unsupported_citations=unsupported,
        )


def _citation_supported(
    citation: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    retrieved_figures: list[dict[str, Any]],
) -> bool:
    chunk_id = citation.get("chunk_id")
    if chunk_id is not None:
        return any(str(chunk.get("chunk_id")) == str(chunk_id) for chunk in retrieved_chunks)

    figure_id = citation.get("figure_id")
    if figure_id is not None:
        return any(str(figure.get("figure_id")) == str(figure_id) for figure in retrieved_figures)

    paper_id = citation.get("paper_id")
    page = citation.get("page")
    if paper_id is None or page is None:
        return False

    return any(
        str(chunk.get("paper_id")) == str(paper_id) and chunk.get("page") == page
        for chunk in retrieved_chunks
    ) or any(
        str(figure.get("paper_id")) == str(paper_id) and figure.get("page") == page
        for figure in retrieved_figures
    )


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _evaluation_result(
    passed: bool,
    reason: str,
    citation_count: int,
    retrieved_chunk_count: int,
    retrieved_figure_count: int,
    unsupported_citations: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    return {
        "metric_name": "answer_citation_support",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "details": {
            "status": "valid" if passed else "invalid",
            "reason": reason,
            "citation_count": citation_count,
            "retrieved_chunk_count": retrieved_chunk_count,
            "retrieved_figure_count": retrieved_figure_count,
            "unsupported_citations": unsupported_citations or [],
        },
    }

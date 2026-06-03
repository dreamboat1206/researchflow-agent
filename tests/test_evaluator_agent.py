from __future__ import annotations

from agents.evaluator_agent import EvaluatorAgent


def test_answer_without_citation_is_invalid_when_context_exists() -> None:
    result = EvaluatorAgent().evaluate_answer(
        answer="ZoomDet improves UAV detection.",
        citations=[],
        retrieved_chunks=[
            {
                "paper_id": 1,
                "chunk_id": "1-p2-c1",
                "page": 2,
                "chunk_text": "ZoomDet improves UAV detection.",
            }
        ],
    )

    assert result["passed"] is False
    assert result["details"]["status"] == "invalid"
    assert result["details"]["reason"] == "missing_citations"


def test_chunk_citation_from_retrieved_chunk_is_valid() -> None:
    result = EvaluatorAgent().evaluate_answer(
        answer="ZoomDet improves UAV detection. [1]",
        citations=[{"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"}],
        retrieved_chunks=[
            {
                "paper_id": 1,
                "chunk_id": "1-p2-c1",
                "page": 2,
                "chunk_text": "ZoomDet improves UAV detection.",
            }
        ],
    )

    assert result["passed"] is True
    assert result["details"]["status"] == "valid"
    assert result["details"]["reason"] == "citations_supported"


def test_figure_citation_from_retrieved_figure_is_valid() -> None:
    result = EvaluatorAgent().evaluate_answer(
        answer="The figure shows an architecture. Reference: figure_id=1_fig_2_1",
        citations=[{"paper_id": 1, "figure_id": "1_fig_2_1", "page": 2}],
        retrieved_figures=[
            {
                "paper_id": 1,
                "figure_id": "1_fig_2_1",
                "page": 2,
                "caption": "Figure 1: Architecture.",
            }
        ],
    )

    assert result["passed"] is True
    assert result["details"]["status"] == "valid"


def test_unsupported_citation_is_invalid() -> None:
    result = EvaluatorAgent().evaluate_answer(
        answer="Unsupported answer. [1]",
        citations=[{"title": "ZoomDet", "page": 3, "chunk_id": "wrong"}],
        retrieved_chunks=[{"paper_id": 1, "chunk_id": "1-p2-c1", "page": 2}],
    )

    assert result["passed"] is False
    assert result["details"]["reason"] == "unsupported_citations"
    assert result["details"]["unsupported_citations"] == [
        {"title": "ZoomDet", "page": 3, "chunk_id": "wrong"}
    ]


def test_no_retrieval_with_refusal_is_valid() -> None:
    result = EvaluatorAgent().evaluate_answer(
        answer="未找到足够依据，无法回答。",
        citations=[],
        retrieved_chunks=[],
        retrieved_figures=[],
    )

    assert result["passed"] is True
    assert result["details"]["reason"] == "no_retrieval_refused"


def test_no_retrieval_without_refusal_is_invalid() -> None:
    result = EvaluatorAgent().evaluate_answer(
        answer="The answer is definitely true.",
        citations=[],
        retrieved_chunks=[],
        retrieved_figures=[],
    )

    assert result["passed"] is False
    assert result["details"]["reason"] == "no_retrieval_but_answered"


def test_evaluate_state_uses_research_state_fields() -> None:
    result = EvaluatorAgent().evaluate_state(
        {
            "final_answer": "The figure shows an architecture.",
            "citations": [{"paper_id": 1, "figure_id": "1_fig_2_1", "page": 2}],
            "retrieved_figures": [{"paper_id": 1, "figure_id": "1_fig_2_1", "page": 2}],
        }
    )

    assert result["passed"] is True

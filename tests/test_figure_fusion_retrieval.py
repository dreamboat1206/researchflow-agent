from __future__ import annotations

from agents.figure_retrieval_agent import FigureRetrievalAgent


class FakeFigureTextStore:
    def search_figures_text(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "figure_id": "fig_a",
                "paper_id": 1,
                "page": 2,
                "image_path": "data/figures/1/fig_a.png",
                "caption": "Transformer architecture diagram.",
                "nearby_text": "Encoder decoder attention blocks.",
                "figure_type": "architecture",
                "score": 0.8,
            },
            {
                "figure_id": "fig_b",
                "paper_id": 1,
                "page": 3,
                "image_path": "data/figures/1/fig_b.png",
                "caption": "Training result chart.",
                "nearby_text": "Accuracy and latency comparison.",
                "figure_type": "chart",
                "score": 0.6,
            },
        ][:top_k]


class FakeFigureImageStore:
    def search_figures_by_image_text(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "figure_id": "fig_b",
                "paper_id": 1,
                "page": 3,
                "image_path": "data/figures/1/fig_b.png",
                "caption": "Training result chart.",
                "figure_type": "chart",
                "score": 0.9,
            },
            {
                "figure_id": "fig_a",
                "paper_id": 1,
                "page": 2,
                "image_path": "data/figures/1/fig_a.png",
                "caption": "Transformer architecture diagram.",
                "figure_type": "architecture",
                "score": 0.3,
            },
        ][:top_k]


def test_search_figures_fusion_normalizes_deduplicates_and_sorts() -> None:
    agent = FigureRetrievalAgent(
        figure_store=FakeFigureTextStore(),
        figure_image_store=FakeFigureImageStore(),
    )

    results = agent.search_figures("Transformer architecture", top_k=2, mode="fusion")

    assert [result["figure_id"] for result in results] == ["fig_a", "fig_b"]
    assert results[0]["final_score"] == 0.7
    assert results[0]["score_breakdown"] == {
        "caption_text_score": 1.0,
        "image_score": 0.0,
        "nearby_text_score": 1.0,
        "weights": {
            "caption_text_score": 0.5,
            "image_score": 0.3,
            "nearby_text_score": 0.2,
        },
    }
    assert results[1]["final_score"] == 0.3
    assert results[1]["score_breakdown"]["image_score"] == 1.0


def test_search_figures_fusion_skips_empty_query() -> None:
    agent = FigureRetrievalAgent(
        figure_store=FakeFigureTextStore(),
        figure_image_store=FakeFigureImageStore(),
    )

    assert agent.search_figures("   ", top_k=5, mode="fusion") == []

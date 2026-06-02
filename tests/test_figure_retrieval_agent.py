from __future__ import annotations

from agents.figure_retrieval_agent import FigureRetrievalAgent


class FakeFigureStore:
    def search_figures_text(self, query: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": f"Figure for {query}",
                "nearby_text": "Transformer encoder decoder blocks.",
                "figure_type": "architecture",
                "score": 0.89,
            }
        ][:top_k]


def test_search_figures_by_text_formats_qdrant_hits() -> None:
    agent = FigureRetrievalAgent(figure_store=FakeFigureStore())

    results = agent.search_figures_by_text("Transformer architecture", top_k=1)

    assert results == [
        {
            "figure_id": "7_fig_2_1",
            "paper_id": 7,
            "page": 2,
            "image_path": "data/figures/7/7_fig_2_1.png",
            "caption": "Figure for Transformer architecture",
            "nearby_text": "Transformer encoder decoder blocks.",
            "figure_type": "architecture",
            "score": 0.89,
        }
    ]


def test_search_figures_by_text_skips_empty_query() -> None:
    agent = FigureRetrievalAgent(figure_store=FakeFigureStore())

    assert agent.search_figures_by_text("   ", top_k=5) == []

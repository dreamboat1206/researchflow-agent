from __future__ import annotations

from graph.figure_ingest_graph import invoke_figure_ingest


def fake_extract_figures_from_pdf(
    file_path,
    paper_id,
    config_path,
    min_width,
    min_height,
    write_to_sqlite,
):
    return [
        {
            "figure_id": f"{paper_id}_fig_1_1",
            "paper_id": paper_id,
            "page": 1,
            "figure_index": 1,
            "image_path": "data/figures/1/1_fig_1_1.png",
            "caption": "Figure 1: Architecture.",
            "nearby_text": "Transformer blocks.",
            "figure_type": "architecture",
        }
    ]


class FakeFigureTextStore:
    collection_name = "paper_figures_text"

    def __init__(self, embedding_model, config_path):
        self.embedding_model = embedding_model
        self.config_path = config_path

    def create_collection(self) -> None:
        return None

    def upsert_figures_text(self, figures) -> int:
        return len(figures)


class FakeFigureImageStore:
    collection_name = "paper_figures_image"

    def __init__(self, embedding_model, config_path):
        self.embedding_model = embedding_model
        self.config_path = config_path

    def create_collection(self) -> None:
        return None

    def upsert_figures_image(self, figures) -> int:
        return len(figures)


def test_invoke_figure_ingest_graph_extracts_and_upserts_vectors() -> None:
    result = invoke_figure_ingest(
        "paper.pdf",
        paper_id=1,
        config_path="config.yaml",
        extract_figures_func=fake_extract_figures_from_pdf,
        text_embedding_model_factory=lambda config_path: object(),
        figure_text_store_factory=FakeFigureTextStore,
        image_embedding_model_factory=lambda config_path: object(),
        figure_image_store_factory=FakeFigureImageStore,
    )

    assert result["figure_count"] == 1
    assert result["figure_text_vector_count"] == 1
    assert result["figure_image_vector_count"] == 1
    assert result["text_collection"] == "paper_figures_text"
    assert result["image_collection"] == "paper_figures_image"

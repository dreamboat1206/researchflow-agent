from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

from storage.qdrant_store import (
    QdrantFigureImageStore,
    QdrantFigureTextStore,
    QdrantTextStore,
    _create_client,
    _ensure_no_proxy_for_local_qdrant,
    get_collection_name,
    get_figures_collection_name,
    get_figures_image_collection_name,
)


class FakeEmbeddingModel:
    def embedding_dimension(self) -> int:
        return 3

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.5, 1.0] for text in texts]

    def encode_query(self, query: str) -> list[float]:
        return [float(len(query)), 0.5, 1.0]


class FakeImageEmbeddingModel:
    def embedding_dimension(self) -> int:
        return 4

    def encode_image(self, image_path: Path) -> list[float]:
        return [float(len(str(image_path))), 1.0, 0.5, 0.25]

    def encode_text(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5, 0.25]


class FakeQdrantClient:
    def __init__(self, collection_exists: bool = False):
        self._collection_exists = collection_exists
        self.created_collections: list[dict[str, object]] = []
        self.upserted_points: list[object] = []
        self.search_calls: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self._collection_exists

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.created_collections.append(
            {"collection_name": collection_name, "vectors_config": vectors_config}
        )
        self._collection_exists = True

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upserted_points.extend(points)

    def search(self, collection_name: str, query_vector: list[float], limit: int) -> list[object]:
        self.search_calls.append(
            {"collection_name": collection_name, "query_vector": query_vector, "limit": limit}
        )
        return [
            SimpleNamespace(
                id="point-1",
                score=0.93,
                payload={
                    "paper_id": 7,
                    "chunk_id": "7-p1-c1",
                    "page": 1,
                    "text": "Transformer text",
                },
            )
        ]


def test_get_collection_name_reads_config() -> None:
    test_dir = Path("data/test_qdrant_store")
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / f"config-{uuid.uuid4().hex}.yaml"
    config_path.write_text(
        "qdrant:\n"
        "  collection_name: papers_text\n",
        encoding="utf-8",
    )

    assert get_collection_name(config_path) == "papers_text"


def test_get_figures_collection_name_reads_config() -> None:
    test_dir = Path("data/test_qdrant_store")
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / f"config-{uuid.uuid4().hex}.yaml"
    config_path.write_text(
        "qdrant:\n"
        "  figures_collection_name: paper_figures_text\n",
        encoding="utf-8",
    )

    assert get_figures_collection_name(config_path) == "paper_figures_text"


def test_get_figures_image_collection_name_reads_config() -> None:
    test_dir = Path("data/test_qdrant_store")
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / f"config-{uuid.uuid4().hex}.yaml"
    config_path.write_text(
        "qdrant:\n"
        "  figures_image_collection_name: paper_figures_image\n",
        encoding="utf-8",
    )

    assert get_figures_image_collection_name(config_path) == "paper_figures_image"


def test_create_client_supports_local_qdrant_path() -> None:
    test_dir = Path("data/test_qdrant_store")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_id = uuid.uuid4().hex
    config_path = test_dir / f"config-{test_id}.yaml"
    qdrant_path = f"qdrant-{test_id}"
    config_path.write_text(
        "qdrant:\n"
        "  mode: local\n"
        f"  path: {qdrant_path}\n"
        "  collection_name: papers_text\n",
        encoding="utf-8",
    )

    client = _create_client(config_path)

    assert (test_dir / qdrant_path).exists()
    client.close()


def test_local_qdrant_url_bypasses_proxy(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "example.com")

    _ensure_no_proxy_for_local_qdrant("http://localhost:6333")

    no_proxy_hosts = os.environ["NO_PROXY"].split(",")
    assert "example.com" in no_proxy_hosts
    assert "localhost" in no_proxy_hosts
    assert "127.0.0.1" in no_proxy_hosts


def test_create_collection_uses_embedding_dimension() -> None:
    client = FakeQdrantClient(collection_exists=False)
    store = QdrantTextStore(
        collection_name="papers_text",
        embedding_model=FakeEmbeddingModel(),
        client=client,
    )

    store.create_collection()

    assert client.created_collections[0]["collection_name"] == "papers_text"
    vector_config = client.created_collections[0]["vectors_config"]
    assert vector_config.size == 3


def test_upsert_text_chunks_embeds_payloads_without_external_qdrant() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantTextStore(
        collection_name="papers_text",
        embedding_model=FakeEmbeddingModel(),
        client=client,
    )

    count = store.upsert_text_chunks(
        [
            {
                "paper_id": 7,
                "chunk_id": "7-p1-c1",
                "page": 1,
                "chunk_text": "Transformer text",
            }
        ]
    )

    assert count == 1
    point = client.upserted_points[0]
    assert point.vector == [16.0, 0.5, 1.0]
    assert point.payload == {
        "paper_id": 7,
        "chunk_id": "7-p1-c1",
        "page": 1,
        "text": "Transformer text",
    }


def test_search_text_returns_top_k_chunks() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantTextStore(
        collection_name="papers_text",
        embedding_model=FakeEmbeddingModel(),
        client=client,
    )

    results = store.search_text("attention", top_k=3)

    assert client.search_calls == [
        {"collection_name": "papers_text", "query_vector": [9.0, 0.5, 1.0], "limit": 3}
    ]
    assert results == [
        {
            "id": "point-1",
            "score": 0.93,
            "paper_id": 7,
            "chunk_id": "7-p1-c1",
            "page": 1,
            "text": "Transformer text",
            "payload": {
                "paper_id": 7,
                "chunk_id": "7-p1-c1",
                "page": 1,
                "text": "Transformer text",
            },
        }
    ]


def test_upsert_figures_text_embeds_caption_and_nearby_text() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantFigureTextStore(
        collection_name="paper_figures_text",
        embedding_model=FakeEmbeddingModel(),
        client=client,
    )

    count = store.upsert_figures_text(
        [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Figure 1: Transformer architecture.",
                "nearby_text": "The encoder and decoder use attention blocks.",
                "figure_type": "architecture",
            }
        ]
    )

    assert count == 1
    point = client.upserted_points[0]
    assert point.payload == {
        "figure_id": "7_fig_2_1",
        "paper_id": 7,
        "page": 2,
        "image_path": "data/figures/7/7_fig_2_1.png",
        "caption": "Figure 1: Transformer architecture.",
        "nearby_text": "The encoder and decoder use attention blocks.",
        "figure_type": "architecture",
        "text": "Figure 1: Transformer architecture.\nThe encoder and decoder use attention blocks.",
    }


def test_search_figures_text_returns_top_k_figures() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantFigureTextStore(
        collection_name="paper_figures_text",
        embedding_model=FakeEmbeddingModel(),
        client=client,
    )
    client.search = lambda collection_name, query_vector, limit: [
        SimpleNamespace(
            id="figure-point-1",
            score=0.88,
            payload={
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Figure 1: Transformer architecture.",
                "nearby_text": "Attention blocks.",
                "figure_type": "architecture",
                "text": "Figure 1: Transformer architecture.\nAttention blocks.",
            },
        )
    ]

    results = store.search_figures_text("Transformer architecture", top_k=1)

    assert results == [
        {
            "id": "figure-point-1",
            "score": 0.88,
            "figure_id": "7_fig_2_1",
            "paper_id": 7,
            "page": 2,
            "image_path": "data/figures/7/7_fig_2_1.png",
            "caption": "Figure 1: Transformer architecture.",
            "nearby_text": "Attention blocks.",
            "figure_type": "architecture",
            "text": "Figure 1: Transformer architecture.\nAttention blocks.",
            "payload": {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Figure 1: Transformer architecture.",
                "nearby_text": "Attention blocks.",
                "figure_type": "architecture",
                "text": "Figure 1: Transformer architecture.\nAttention blocks.",
            },
        }
    ]


def test_search_figures_text_returns_empty_when_collection_missing() -> None:
    client = FakeQdrantClient(collection_exists=False)
    store = QdrantFigureTextStore(
        collection_name="paper_figures_text",
        embedding_model=FakeEmbeddingModel(),
        client=client,
    )

    assert store.search_figures_text("Transformer architecture", top_k=3) == []


def test_upsert_figures_image_embeds_existing_image(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"fake-image")
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantFigureImageStore(
        collection_name="paper_figures_image",
        embedding_model=FakeImageEmbeddingModel(),
        client=client,
    )

    count = store.upsert_figures_image(
        [
            {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": str(image_path),
                "caption": "Figure 1: Transformer architecture.",
                "figure_type": "architecture",
            }
        ]
    )

    assert count == 1
    point = client.upserted_points[0]
    assert point.vector == [float(len(str(image_path))), 1.0, 0.5, 0.25]
    assert point.payload == {
        "figure_id": "7_fig_2_1",
        "paper_id": 7,
        "page": 2,
        "image_path": str(image_path),
        "caption": "Figure 1: Transformer architecture.",
        "figure_type": "architecture",
    }


def test_search_figures_by_image_text_returns_top_k_figures() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantFigureImageStore(
        collection_name="paper_figures_image",
        embedding_model=FakeImageEmbeddingModel(),
        client=client,
    )
    client.search = lambda collection_name, query_vector, limit: [
        SimpleNamespace(
            id="image-point-1",
            score=0.87,
            payload={
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Figure 1: Transformer architecture.",
                "figure_type": "architecture",
            },
        )
    ]

    results = store.search_figures_by_image_text("Transformer architecture", top_k=1)

    assert results == [
        {
            "id": "image-point-1",
            "score": 0.87,
            "figure_id": "7_fig_2_1",
            "paper_id": 7,
            "page": 2,
            "image_path": "data/figures/7/7_fig_2_1.png",
            "caption": "Figure 1: Transformer architecture.",
            "figure_type": "architecture",
            "payload": {
                "figure_id": "7_fig_2_1",
                "paper_id": 7,
                "page": 2,
                "image_path": "data/figures/7/7_fig_2_1.png",
                "caption": "Figure 1: Transformer architecture.",
                "figure_type": "architecture",
            },
        }
    ]

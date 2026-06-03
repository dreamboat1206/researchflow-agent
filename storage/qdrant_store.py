from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from models.image_embedding import ImageEmbeddingModel
from models.text_embedding import TextEmbeddingModel
from storage.sqlite_store import DEFAULT_CONFIG_PATH, load_config


class QdrantTextStore:
    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: TextEmbeddingModel | None = None,
        client: Any | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.config_path = config_path
        self.collection_name = collection_name or get_collection_name(config_path)
        self.embedding_model = embedding_model or TextEmbeddingModel(config_path=config_path)
        self.client = client or _create_client(config_path)

    def create_collection(self, vector_size: int | None = None) -> None:
        size = vector_size or self.embedding_model.embedding_dimension()
        if _collection_exists(self.client, self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
        )

    def upsert_text_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        embeddings = self.embedding_model.encode_texts([str(chunk["chunk_text"]) for chunk in chunks])
        points = [
            qmodels.PointStruct(
                id=_point_id(str(chunk["chunk_id"])),
                vector=embedding,
                payload=_chunk_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = self.embedding_model.encode_query(query)
        hits = _search(self.client, self.collection_name, query_vector, top_k)
        return [_hit_to_result(hit) for hit in hits]


class QdrantFigureTextStore(QdrantTextStore):
    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: TextEmbeddingModel | None = None,
        client: Any | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        super().__init__(
            collection_name=collection_name or get_figures_collection_name(config_path),
            embedding_model=embedding_model,
            client=client,
            config_path=config_path,
        )

    def upsert_figures_text(self, figures: list[dict[str, Any]]) -> int:
        indexable_figures = [
            figure for figure in figures if figure.get("figure_id") and _figure_text(figure)
        ]
        if not indexable_figures:
            return 0

        embeddings = self.embedding_model.encode_texts(
            [_figure_text(figure) for figure in indexable_figures]
        )
        points = [
            qmodels.PointStruct(
                id=_point_id(f"figure:{figure['figure_id']}"),
                vector=embedding,
                payload=_figure_payload(figure),
            )
            for figure, embedding in zip(indexable_figures, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search_figures_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not _collection_exists(self.client, self.collection_name):
            return []
        query_vector = self.embedding_model.encode_query(query)
        hits = _search(self.client, self.collection_name, query_vector, top_k)
        return [_hit_to_figure_result(hit) for hit in hits]


class QdrantFigureImageStore:
    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: ImageEmbeddingModel | None = None,
        client: Any | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.config_path = config_path
        self.collection_name = collection_name or get_figures_image_collection_name(config_path)
        self.embedding_model = embedding_model or ImageEmbeddingModel(config_path=config_path)
        self.client = client or _create_client(config_path)

    def create_collection(self, vector_size: int | None = None) -> None:
        size = vector_size or self.embedding_model.embedding_dimension()
        if _collection_exists(self.client, self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
        )

    def upsert_figures_image(self, figures: list[dict[str, Any]]) -> int:
        indexable_figures = [
            figure for figure in figures if figure.get("figure_id") and figure.get("image_path")
        ]
        if not indexable_figures:
            return 0

        points = []
        for figure in indexable_figures:
            image_path = _resolve_stored_path(self.config_path, str(figure["image_path"]))
            if not image_path.exists():
                continue
            points.append(
                qmodels.PointStruct(
                    id=_point_id(f"figure-image:{figure['figure_id']}"),
                    vector=self.embedding_model.encode_image(image_path),
                    payload=_figure_image_payload(figure),
                )
            )
        if not points:
            return 0
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search_figures_by_image_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not _collection_exists(self.client, self.collection_name):
            return []
        query_vector = self.embedding_model.encode_text(query)
        hits = _search(self.client, self.collection_name, query_vector, top_k)
        return [_hit_to_figure_image_result(hit) for hit in hits]


def get_collection_name(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = load_config(config_path)
    collection_name = config.get("qdrant", {}).get("collection_name")
    if not collection_name:
        raise ValueError("Missing qdrant.collection_name in config.yaml")
    return str(collection_name)


def get_figures_collection_name(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = load_config(config_path)
    collection_name = config.get("qdrant", {}).get("figures_collection_name")
    if not collection_name:
        return "paper_figures_text"
    return str(collection_name)


def get_figures_image_collection_name(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = load_config(config_path)
    collection_name = config.get("qdrant", {}).get("figures_image_collection_name")
    if not collection_name:
        return "paper_figures_image"
    return str(collection_name)


def _create_client(config_path: str | Path) -> QdrantClient:
    config = load_config(config_path)
    qdrant_config = config.get("qdrant", {})
    mode = str(qdrant_config.get("mode", "")).lower()
    local_path = qdrant_config.get("path")
    if mode == "local" or (not mode and local_path):
        if not local_path:
            raise ValueError("Missing qdrant.path for local Qdrant mode")
        path = Path(str(local_path))
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))

    url = qdrant_config.get("url")
    if url:
        _ensure_no_proxy_for_local_qdrant(str(url))
        return QdrantClient(url=str(url))

    host = qdrant_config.get("host", "localhost")
    port = int(qdrant_config.get("port", 6333))
    if str(host) in {"localhost", "127.0.0.1", "::1"}:
        _merge_no_proxy(["localhost", "127.0.0.1", "::1"])
    return QdrantClient(host=str(host), port=port)


def _collection_exists(client: Any, collection_name: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name))

    collections = client.get_collections().collections
    return any(collection.name == collection_name for collection in collections)


def _ensure_no_proxy_for_local_qdrant(url: str) -> None:
    hostname = urlparse(url).hostname
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        _merge_no_proxy(["localhost", "127.0.0.1", "::1"])


def _merge_no_proxy(hosts: list[str]) -> None:
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        values = [value.strip() for value in existing.split(",") if value.strip()]
        for host in hosts:
            if host not in values:
                values.append(host)
        os.environ[env_name] = ",".join(values)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"researchflow:{chunk_id}"))


def _chunk_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": chunk.get("paper_id"),
        "chunk_id": chunk.get("chunk_id"),
        "page": chunk.get("page"),
        "text": chunk.get("chunk_text"),
    }


def _figure_text(figure: dict[str, Any]) -> str:
    parts = [
        str(figure.get("caption") or "").strip(),
        str(figure.get("nearby_text") or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def _figure_payload(figure: dict[str, Any]) -> dict[str, Any]:
    text = _figure_text(figure)
    return {
        "figure_id": figure.get("figure_id"),
        "paper_id": figure.get("paper_id"),
        "page": figure.get("page") or figure.get("page_number"),
        "image_path": figure.get("image_path"),
        "caption": figure.get("caption"),
        "nearby_text": figure.get("nearby_text"),
        "figure_type": figure.get("figure_type"),
        "text": text,
    }


def _figure_image_payload(figure: dict[str, Any]) -> dict[str, Any]:
    return {
        "figure_id": figure.get("figure_id"),
        "paper_id": figure.get("paper_id"),
        "page": figure.get("page") or figure.get("page_number"),
        "image_path": figure.get("image_path"),
        "caption": figure.get("caption"),
        "figure_type": figure.get("figure_type"),
    }


def _resolve_stored_path(config_path: str | Path, stored_path: str) -> Path:
    path = Path(stored_path)
    if path.exists():
        return path
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent / path


def _search(client: Any, collection_name: str, query_vector: list[float], top_k: int) -> Any:
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return result.points


def _hit_to_result(hit: Any) -> dict[str, Any]:
    payload = dict(getattr(hit, "payload", None) or {})
    return {
        "id": str(getattr(hit, "id", "")),
        "score": float(getattr(hit, "score", 0.0)),
        "paper_id": payload.get("paper_id"),
        "chunk_id": payload.get("chunk_id"),
        "page": payload.get("page"),
        "text": payload.get("text"),
        "payload": payload,
    }


def _hit_to_figure_result(hit: Any) -> dict[str, Any]:
    payload = dict(getattr(hit, "payload", None) or {})
    return {
        "id": str(getattr(hit, "id", "")),
        "score": float(getattr(hit, "score", 0.0)),
        "figure_id": payload.get("figure_id"),
        "paper_id": payload.get("paper_id"),
        "page": payload.get("page"),
        "image_path": payload.get("image_path"),
        "caption": payload.get("caption"),
        "nearby_text": payload.get("nearby_text"),
        "figure_type": payload.get("figure_type"),
        "text": payload.get("text"),
        "payload": payload,
    }


def _hit_to_figure_image_result(hit: Any) -> dict[str, Any]:
    payload = dict(getattr(hit, "payload", None) or {})
    return {
        "id": str(getattr(hit, "id", "")),
        "score": float(getattr(hit, "score", 0.0)),
        "figure_id": payload.get("figure_id"),
        "paper_id": payload.get("paper_id"),
        "page": payload.get("page"),
        "image_path": payload.get("image_path"),
        "caption": payload.get("caption"),
        "figure_type": payload.get("figure_type"),
        "payload": payload,
    }

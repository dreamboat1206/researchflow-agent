from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from storage.qdrant_store import QdrantTextStore
from storage.sqlite_store import DEFAULT_CONFIG_PATH, get_db_path, load_config


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/config")
def system_config() -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIG_PATH)
    return {
        "database": {"path": str(get_db_path(DEFAULT_CONFIG_PATH))},
        "qdrant": {
            "url": config.get("qdrant", {}).get("url"),
            "collection_name": config.get("qdrant", {}).get("collection_name"),
            "figures_collection_name": config.get("qdrant", {}).get("figures_collection_name"),
            "figures_image_collection_name": config.get("qdrant", {}).get("figures_image_collection_name"),
        },
        "models": {
            "embedding_model": config.get("models", {}).get("embedding_model"),
            "image_embedding_model": config.get("models", {}).get("image_embedding_model"),
        },
    }


@router.get("/qdrant")
def qdrant_status() -> dict[str, Any]:
    try:
        store = QdrantTextStore()
        collections = store.client.get_collections()
        names = [collection.name for collection in collections.collections]
        return {
            "connected": True,
            "collection_name": store.collection_name,
            "collections": names,
        }
    except Exception as error:
        return {
            "connected": False,
            "collection_name": None,
            "collections": [],
            "error": str(error),
        }

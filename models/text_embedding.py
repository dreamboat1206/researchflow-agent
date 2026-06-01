from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from storage.sqlite_store import DEFAULT_CONFIG_PATH, load_config


class TextEmbeddingModel:
    def __init__(self, model_name: str | None = None, config_path: str | Path = DEFAULT_CONFIG_PATH):
        config = load_config(config_path)
        self.model_name = model_name or get_embedding_model_name(config_path)
        self.local_files_only = bool(config.get("models", {}).get("embedding_local_files_only", False))
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        return self._model

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]

    def encode_query(self, query: str) -> list[float]:
        return self.encode_texts([query])[0]

    def embedding_dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        if dimension is not None:
            return int(dimension)
        return len(self.encode_query("dimension probe"))


def get_embedding_model_name(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = load_config(config_path)
    model_name = config.get("models", {}).get("embedding_model")
    if not model_name:
        raise ValueError("Missing models.embedding_model in config.yaml")
    return str(model_name)

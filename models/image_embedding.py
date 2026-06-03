from __future__ import annotations

from pathlib import Path
from typing import Any

from storage.sqlite_store import DEFAULT_CONFIG_PATH, load_config


class ImageEmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | None = None,
    ):
        config = load_config(config_path)
        model_config = config.get("models", {})
        self.model_name = model_name or str(model_config.get("image_embedding_model") or "ViT-B-32")
        self.pretrained = pretrained or str(
            model_config.get("image_embedding_pretrained") or "laion2b_s34b_b79k"
        )
        self.device = device or str(model_config.get("image_embedding_device") or "cpu")
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None

    def encode_image(self, image_path: str | Path) -> list[float]:
        self._ensure_loaded()
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].detach().cpu().tolist()

    def encode_text(self, text: str) -> list[float]:
        self._ensure_loaded()
        import torch

        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].detach().cpu().tolist()

    def embedding_dimension(self) -> int:
        return len(self.encode_text("dimension probe"))

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

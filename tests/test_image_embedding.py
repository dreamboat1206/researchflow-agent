from __future__ import annotations

import uuid
from pathlib import Path

from models.image_embedding import ImageEmbeddingModel


def test_image_embedding_model_reads_config() -> None:
    test_dir = Path("data/test_image_embedding")
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / f"config-{uuid.uuid4().hex}.yaml"
    config_path.write_text(
        "models:\n"
        "  image_embedding_model: ViT-B-16\n"
        "  image_embedding_pretrained: test-pretrained\n"
        "  image_embedding_device: cpu\n",
        encoding="utf-8",
    )

    model = ImageEmbeddingModel(config_path=config_path)

    assert model.model_name == "ViT-B-16"
    assert model.pretrained == "test-pretrained"
    assert model.device == "cpu"

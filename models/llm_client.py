from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from storage.sqlite_store import DEFAULT_CONFIG_PATH, load_config


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        config = load_config(config_path)
        llm_config = config.get("llm", {})
        model_config = config.get("models", {})
        self.provider = provider or str(llm_config.get("provider", "mock"))
        self.model_name = model_name or str(model_config.get("llm_model", "mock-llm"))
        self.base_url = base_url or llm_config.get("base_url")
        self.api_key = api_key or llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    def generate(self, prompt: str) -> str:
        if self.provider == "mock" or not self.base_url or not self.api_key:
            return self._mock_generate(prompt)
        return self._openai_compatible_generate(prompt)

    def _openai_compatible_generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only with the provided paper context and cite sources.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        url = f"{str(self.base_url).rstrip('/')}/chat/completions"
        response = httpx.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _mock_generate(self, prompt: str) -> str:
        context_preview = _first_context_line(prompt)
        return f"Mock answer based on retrieved paper context. {context_preview}".strip()


def _first_context_line(prompt: str) -> str:
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("[1]"):
            return stripped
    return ""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiResult:
    ok: bool
    data: Any = None
    error: str | None = None
    status_code: int | None = None


class ResearchFlowApiClient:
    def __init__(self, api_url: str | None = None, timeout: float = 30.0, upload_timeout: float = 300.0):
        self.api_url = (api_url or os.environ.get("API_URL") or "http://localhost:8000").rstrip("/")
        self.timeout = timeout
        self.upload_timeout = upload_timeout

    def health(self) -> ApiResult:
        return self._request("GET", "/health")

    def stats(self) -> ApiResult:
        return self._request("GET", "/stats")

    def list_papers(self, filters: dict[str, Any] | None = None) -> ApiResult:
        return self._request("GET", "/papers", params=_clean_params(filters or {}))

    def get_paper(self, paper_id: int | str) -> ApiResult:
        return self._request("GET", f"/papers/{paper_id}")

    def get_paper_chunks(self, paper_id: int | str, limit: int | None = None) -> ApiResult:
        return self._request("GET", f"/papers/{paper_id}/chunks", params=_clean_params({"limit": limit}))

    def ingest_paper(self, file: Any, auto_organize: bool = True, organize_mode: str = "copy") -> ApiResult:
        files = {"file": (getattr(file, "name", "paper.pdf"), file, "application/pdf")}
        params = {"auto_organize": auto_organize, "organize_mode": organize_mode}
        return self._request("POST", "/papers/ingest", params=params, files=files, timeout=self.upload_timeout)

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_id: int | str | None = None,
        collection_id: str | None = None,
    ) -> ApiResult:
        result = self._request("POST", "/search/text", json={"query": query, "top_k": top_k})
        if result.ok and score_threshold is not None:
            results = result.data.get("results", [])
            result.data["results"] = [item for item in results if (item.get("score") or 0) >= score_threshold]
        return result

    def ask(
        self,
        question: str,
        top_k: int = 5,
        paper_id: int | str | None = None,
        collection_id: str | None = None,
        show_chunks: bool = True,
    ) -> ApiResult:
        return self._request("POST", "/qa/paper", json={"question": question, "top_k": top_k})

    def organize_paper(self, paper_id: int | str, dry_run: bool = True, mode: str = "copy") -> ApiResult:
        return self._request("POST", f"/organizer/papers/{paper_id}", json={"dry_run": dry_run, "mode": mode})

    def organize_papers(self, dry_run: bool = True, mode: str = "copy", limit: int | None = None) -> ApiResult:
        return self._request("POST", "/organizer/papers", json={"dry_run": dry_run, "mode": mode, "limit": limit})

    def list_tags(self, paper_id: int | str | None = None, tag_type: str | None = None) -> ApiResult:
        return self._request("GET", "/organizer/tags", params=_clean_params({"paper_id": paper_id, "tag_type": tag_type}))

    def list_collections(self) -> ApiResult:
        return self._request("GET", "/organizer/collections")

    def qdrant_status(self) -> ApiResult:
        return self._request("GET", "/system/qdrant")

    def system_config(self) -> ApiResult:
        return self._request("GET", "/system/config")

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(method, f"{self.api_url}{path}", **kwargs)
            if response.status_code >= 400:
                return ApiResult(False, error=_error_message(response), status_code=response.status_code)
            return ApiResult(True, data=response.json(), status_code=response.status_code)
        except httpx.HTTPError as error:
            return ApiResult(False, error=str(error))


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value not in (None, "")}


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return str(detail or payload)

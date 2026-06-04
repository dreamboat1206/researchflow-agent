from __future__ import annotations

import os
from typing import Any

import streamlit as st


DEFAULT_API_URL = "http://localhost:8000"


def default_api_url() -> str:
    return os.environ.get("API_URL") or _legacy_api_url() or DEFAULT_API_URL


def init_state() -> None:
    defaults: dict[str, Any] = {
        "selected_paper_id": None,
        "selected_collection_id": None,
        "reader_selected_page": 1,
        "reader_selected_chunk_id": None,
        "last_search_query": "",
        "last_search_results": [],
        "last_qa_question": "",
        "last_qa_result": None,
        "api_url": default_api_url(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def init_session_state() -> None:
    init_state()


def select_paper(paper_id: int | str | None) -> None:
    st.session_state["selected_paper_id"] = int(paper_id) if str(paper_id or "").isdigit() else paper_id


def _legacy_api_url() -> str | None:
    host = os.environ.get("API_HOST")
    port = os.environ.get("API_PORT", "8000")
    if not host:
        return None
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{port}"

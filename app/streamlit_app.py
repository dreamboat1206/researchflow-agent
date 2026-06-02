from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st


def _api_base_url() -> str:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = os.environ.get("API_PORT", "8000")
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{port}"


def _format_score(score: object) -> str:
    if isinstance(score, int | float):
        return f"{score:.4f}"
    return "-"


def _figure_query_params(paper_id_text: str, title_query: str) -> dict[str, str | int]:
    params: dict[str, str | int] = {}
    if paper_id_text.strip():
        params["paper_id"] = int(paper_id_text.strip())
    if title_query.strip():
        params["title"] = title_query.strip()
    return params


def _figure_image_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.exists():
        return str(path)
    if not path.is_absolute():
        candidate = Path.cwd() / path
        if candidate.exists():
            return str(candidate)
    container_prefix = "/app/"
    if image_path.startswith(container_prefix):
        candidate = Path(image_path[len(container_prefix) :])
        if candidate.exists():
            return str(candidate)
    normalized_path = image_path.replace("\\", "/")
    data_index = normalized_path.lower().find("data/")
    if data_index >= 0:
        candidate = Path(normalized_path[data_index:])
        if candidate.exists():
            return str(candidate)
    return image_path


def _load_figures(api_base_url: str, params: dict[str, str | int]) -> list[dict[str, Any]]:
    response = httpx.get(f"{api_base_url}/figures", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="ResearchFlow-Agent",
    layout="wide",
)

st.title("ResearchFlow-Agent")
st.caption("A minimal workspace for research document workflows.")

api_base_url = _api_base_url()

home_tab, search_tab, qa_tab, figures_tab = st.tabs(
    ["Overview", "Paper Search", "Paper QA", "Figure Gallery"]
)

with home_tab:
    st.write("Project scaffold is ready. Use the API health check to verify the backend.")
    if st.button("Check local API path"):
        st.code(f"GET {api_base_url}/health")

with search_tab:
    with st.form("paper-search-form"):
        query = st.text_input("Search text", placeholder="Enter a paper topic, method, or keyword")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search")

    if submitted:
        if not query.strip():
            st.warning("Please enter search text.")
        else:
            try:
                response = httpx.post(
                    f"{api_base_url}/search/text",
                    json={"query": query, "top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
                results = response.json()["results"]
            except httpx.HTTPError as exc:
                st.error(f"Search request failed: {exc}")
            else:
                if not results:
                    st.info("No relevant chunks found.")
                for result in results:
                    title = result.get("title") or f"Paper {result.get('paper_id')}"
                    page = result.get("page") or "-"
                    with st.container(border=True):
                        st.subheader(title)
                        st.caption(f"page: {page} | score: {_format_score(result.get('score'))}")
                        st.write(result.get("chunk_text") or "")

with qa_tab:
    with st.form("paper-qa-form"):
        question = st.text_input("Question", placeholder="Ask a question about ingested papers")
        qa_top_k = st.slider("Context chunks", min_value=1, max_value=20, value=5)
        qa_submitted = st.form_submit_button("Ask")

    if qa_submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                response = httpx.post(
                    f"{api_base_url}/qa/paper",
                    json={"question": question, "top_k": qa_top_k},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"QA request failed: {exc}")
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Citations")
                citations = result.get("citations", [])
                if not citations:
                    st.info("No citations returned.")
                for citation in citations:
                    title = citation.get("title") or "Unknown paper"
                    page = citation.get("page") or "-"
                    chunk_id = citation.get("chunk_id") or "-"
                    st.caption(f"{title} | page: {page} | chunk_id: {chunk_id}")

with figures_tab:
    st.subheader("Figure Gallery / 图表浏览")
    with st.form("figure-filter-form"):
        filter_cols = st.columns([1, 3, 1])
        paper_id_text = filter_cols[0].text_input("paper_id", placeholder="1")
        title_query = filter_cols[1].text_input("Paper title", placeholder="ZoomDet")
        gallery_submitted = filter_cols[2].form_submit_button("Load")

    if gallery_submitted:
        try:
            figures = _load_figures(api_base_url, _figure_query_params(paper_id_text, title_query))
        except ValueError:
            st.error("paper_id must be a number.")
        except httpx.HTTPError as exc:
            st.error(f"Figure request failed: {exc}")
        else:
            if not figures:
                st.info("No figures found. Extract figures first, then reload the gallery.")
            for figure in figures:
                title = figure.get("paper_title") or f"Paper {figure.get('paper_id')}"
                caption = figure.get("caption") or "No caption matched yet."
                with st.container(border=True):
                    image_col, detail_col = st.columns([1, 2])
                    image_path = _figure_image_path(figure.get("image_path"))
                    with image_col:
                        if image_path and Path(image_path).exists():
                            st.image(image_path, use_container_width=True)
                        else:
                            st.code(image_path or "No image path")
                    with detail_col:
                        st.subheader(figure.get("figure_id") or "Figure")
                        st.caption(
                            f"{title} | page: {figure.get('page') or '-'} | "
                            f"type: {figure.get('figure_type') or 'other'}"
                        )
                        st.write(caption)
                        with st.expander("Details"):
                            st.write(f"image_path: {figure.get('image_path') or '-'}")
                            st.write(f"paper_id: {figure.get('paper_id')}")
                            st.write(f"page: {figure.get('page') or '-'}")
                            st.write(f"caption: {figure.get('caption') or '-'}")
                            st.write(f"nearby_text: {figure.get('nearby_text') or '-'}")

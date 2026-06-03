from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import streamlit as st

from observability.trace_logger import list_recent_traces


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

home_tab, search_tab, figure_search_tab, qa_tab, figures_tab, trace_tab = st.tabs(
    ["Overview", "Paper Search", "Figure Search", "Paper QA", "Figure Gallery", "Trace Viewer"]
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

with figure_search_tab:
    with st.form("figure-search-form"):
        figure_search_mode = st.selectbox(
            "Search mode",
            ["Fusion", "Caption + nearby text", "Image embedding"],
        )
        figure_query = st.text_input(
            "Figure search text",
            placeholder="Find Transformer architecture figures",
        )
        figure_top_k = st.slider("Number of figures", min_value=1, max_value=20, value=5)
        figure_submitted = st.form_submit_button("Search Figures")

    if figure_submitted:
        if not figure_query.strip():
            st.warning("Please enter figure search text.")
        else:
            endpoint = "/search/figures/image-text" if figure_search_mode == "Image embedding" else "/search/figures"
            payload = {"query": figure_query, "top_k": figure_top_k}
            if figure_search_mode == "Fusion":
                payload["mode"] = "fusion"
            try:
                response = httpx.post(
                    f"{api_base_url}{endpoint}",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                results = response.json()["results"]
            except httpx.HTTPError as exc:
                st.error(f"Figure search request failed: {exc}")
            else:
                if not results:
                    st.info("No relevant figures found.")
                for result in results:
                    with st.container(border=True):
                        image_col, detail_col = st.columns([1, 2])
                        image_path = _figure_image_path(result.get("image_path"))
                        with image_col:
                            if image_path and Path(image_path).exists():
                                st.image(image_path, use_container_width=True)
                            else:
                                st.code(image_path or "No image path")
                        with detail_col:
                            st.subheader(result.get("figure_id") or "Figure")
                            st.caption(
                                f"paper_id: {result.get('paper_id') or '-'} | "
                                f"page: {result.get('page') or '-'} | "
                                f"type: {result.get('figure_type') or 'other'} | "
                                f"score: {_format_score(result.get('final_score') or result.get('score'))}"
                            )
                            st.write(result.get("caption") or "No caption matched yet.")
                            if result.get("score_breakdown"):
                                scores = result["score_breakdown"]
                                st.caption(
                                    "caption: "
                                    f"{_format_score(scores.get('caption_text_score'))} | "
                                    f"image: {_format_score(scores.get('image_score'))} | "
                                    f"nearby: {_format_score(scores.get('nearby_text_score'))} | "
                                    f"final: {_format_score(result.get('final_score'))}"
                                )
                            with st.expander("Nearby text"):
                                st.write(result.get("nearby_text") or "-")

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

with trace_tab:
    st.subheader("Trace Viewer")
    trace_limit = st.slider("Recent traces", min_value=5, max_value=100, value=20, step=5)
    if st.button("Refresh traces"):
        st.rerun()

    try:
        traces = list_recent_traces(limit=trace_limit)
    except Exception as exc:
        st.error(f"Failed to load traces: {exc}")
    else:
        if not traces:
            st.info("No traces recorded yet. Run a QA or search request first.")
        for trace in traces:
            success = "success" if trace.get("success") else "failed"
            title = (
                f"{trace.get('task_type') or 'unknown'} | {success} | "
                f"{trace.get('latency_ms')} ms"
            )
            with st.expander(title):
                st.caption(f"trace_id: {trace.get('trace_id')}")
                st.write(f"query: {trace.get('query') or '-'}")
                if trace.get("error_message"):
                    st.error(trace["error_message"])
                retrieved_items = trace.get("retrieved_items") or []
                st.write(f"retrieved_items: {len(retrieved_items)}")
                if trace.get("final_answer"):
                    st.write(trace["final_answer"])
                st.json(trace)

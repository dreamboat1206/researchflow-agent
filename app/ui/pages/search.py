from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.chunk_viewer import chunk_viewer
from app.ui.state import select_paper


def render(client: ResearchFlowApiClient) -> None:
    st.header("Search")
    st.caption("Semantic search returns relevant chunks, not a final answer.")
    default_query = st.session_state.get("last_search_query", "")
    query = st.text_input("Query", value=default_query)
    top_k = st.slider("top_k", 1, 20, 5)
    threshold = st.slider("score_threshold", 0.0, 1.0, 0.0, 0.01)
    scope = st.selectbox("Scope", ["All papers", "Selected paper", "Selected collection"])
    if scope == "Selected paper" and not st.session_state.get("selected_paper_id"):
        st.warning("Select a paper from Library first.")

    if st.button("Search", type="primary"):
        result = client.search_text(query, top_k=top_k, score_threshold=threshold)
        if not result.ok:
            st.error(result.error or "Search failed.")
            return
        st.session_state["last_search_query"] = query
        st.session_state["last_search_results"] = result.data.get("results", [])

    results = st.session_state.get("last_search_results") or []
    if not results:
        st.info("没有找到足够相关的论文片段。")
    for rank, chunk in enumerate(results, start=1):
        chunk_viewer(chunk, rank=rank)
        if st.button("View Paper", key=f"view-paper-{rank}-{chunk.get('chunk_id')}"):
            select_paper(chunk.get("paper_id"))

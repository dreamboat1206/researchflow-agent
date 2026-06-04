from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.chunk_viewer import chunk_viewer


def render(client: ResearchFlowApiClient) -> None:
    paper_id = st.session_state.get("selected_paper_id")
    if not paper_id:
        st.info("Select a paper from Library first.")
        return

    detail = client.get_paper(paper_id)
    if not detail.ok:
        st.error(detail.error or "Failed to load paper.")
        return
    paper = detail.data.get("paper", {})
    st.subheader(paper.get("title") or f"Paper {paper_id}")
    st.caption(f"paper_id: {paper.get('id')} | year: {paper.get('year') or '-'}")

    page = st.number_input(
        "Reader page",
        min_value=1,
        value=int(st.session_state.get("reader_selected_page") or 1),
    )
    st.session_state["reader_selected_page"] = int(page)

    chunks = client.get_paper_chunks(paper_id, limit=100)
    if not chunks.ok:
        st.error(chunks.error or "Failed to load chunks.")
        return

    page_chunks = [chunk for chunk in chunks.data.get("chunks", []) if int(chunk.get("page") or 0) == int(page)]
    if not page_chunks:
        st.info("No chunks found for this page yet.")
        return

    selected_chunk_id = st.selectbox(
        "Chunk",
        [chunk.get("chunk_id") for chunk in page_chunks],
    )
    st.session_state["reader_selected_chunk_id"] = selected_chunk_id

    for index, chunk in enumerate(page_chunks, start=1):
        chunk_viewer(chunk, rank=index)

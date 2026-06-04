from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.chunk_viewer import chunk_viewer


def render(client: ResearchFlowApiClient) -> None:
    st.header("Paper Detail")
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
    st.caption(
        f"paper_id: {paper.get('id')} | type: {paper.get('paper_type') or '-'} | "
        f"topic: {paper.get('primary_topic') or '-'} | year: {paper.get('year') or '-'}"
    )
    st.write(f"original_path: {paper.get('original_path') or paper.get('source_path') or '-'}")
    st.write(f"organized_path: {paper.get('organized_path') or '-'}")
    st.write(f"organization_status: {paper.get('organization_status') or 'pending'}")
    st.subheader("Tags")
    st.json(detail.data.get("tags") or [])
    st.subheader("Collections")
    st.json(detail.data.get("collections") or [])

    cols = st.columns(2)
    if cols[0].button("Re-organize dry-run"):
        result = client.organize_paper(paper_id, dry_run=True, mode="copy")
        _show_organize_result(result)
    if cols[1].button("Apply organize copy"):
        result = client.organize_paper(paper_id, dry_run=False, mode="copy")
        _show_organize_result(result)

    limit = st.slider("Chunk limit", 1, 100, 10)
    chunks = client.get_paper_chunks(paper_id, limit=limit)
    st.subheader("Chunks")
    if chunks.ok:
        for index, chunk in enumerate(chunks.data.get("chunks", []), start=1):
            chunk_viewer(chunk, rank=index)
    else:
        st.error(chunks.error or "Failed to load chunks.")


def _show_organize_result(result) -> None:
    if not result.ok:
        st.error(result.error or "Organizer request failed.")
        return
    st.success("Organizer request complete.")
    st.json(result.data)

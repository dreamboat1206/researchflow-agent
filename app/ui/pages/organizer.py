from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient


def render(client: ResearchFlowApiClient) -> None:
    st.header("Organizer")
    papers = client.list_papers({"limit": 200})
    if not papers.ok:
        st.error(papers.error or "Failed to load papers.")
        return
    paper_rows = papers.data.get("papers", [])
    scope = st.radio("Scope", ["One paper", "All papers"], horizontal=True)
    mode = st.selectbox("Mode", ["copy", "move"])
    if mode == "move":
        st.warning("Move mode will move the original PDF file. Use only after confirming target paths.")
    limit = st.number_input("Batch limit", min_value=1, max_value=100, value=10)

    selected_paper = None
    if scope == "One paper":
        selected_paper = st.selectbox("Paper", paper_rows, format_func=lambda paper: f"{paper['id']} - {paper['title']}")

    cols = st.columns(2)
    if cols[0].button("Preview Organization"):
        _show_organize_result(_organize(client, scope, selected_paper, dry_run=True, mode=mode, limit=limit))
    if cols[1].button("Apply Organization"):
        st.warning("Confirm target paths before applying. Default behavior is copy.")
        _show_organize_result(_organize(client, scope, selected_paper, dry_run=False, mode=mode, limit=limit))


def _organize(client: ResearchFlowApiClient, scope: str, selected_paper: dict | None, dry_run: bool, mode: str, limit: int):
    if scope == "One paper" and selected_paper:
        result = client.organize_paper(selected_paper["id"], dry_run=dry_run, mode=mode)
    else:
        result = client.organize_papers(dry_run=dry_run, mode=mode, limit=limit)
    return result


def _show_organize_result(result) -> None:
    if not result.ok:
        st.error(result.error or "Organizer request failed.")
        return
    st.success("Organizer request complete.")
    st.json(result.data)

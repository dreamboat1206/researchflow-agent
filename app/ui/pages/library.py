from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.paper_table import paper_table
from app.ui.state import select_paper


def render(client: ResearchFlowApiClient) -> None:
    st.header("Library")
    papers_result = client.list_papers({"limit": 200})
    tags_result = client.list_tags()
    if not papers_result.ok:
        st.error(papers_result.error or "Failed to load papers.")
        return
    papers = papers_result.data.get("papers", [])
    tags = tags_result.data.get("tags", []) if tags_result.ok else []

    with st.expander("Filters", expanded=True):
        title_query = st.text_input("Title contains")
        topic_values = [""] + sorted({paper.get("primary_topic") for paper in papers if paper.get("primary_topic")})
        year_values = [""] + sorted({str(paper.get("year")) for paper in papers if paper.get("year")})
        tag_values = [""] + sorted({tag.get("tag") for tag in tags if tag.get("tag")})
        selected_topic = st.selectbox("primary_topic", topic_values)
        selected_year = st.selectbox("year", year_values)
        selected_tag = st.selectbox("tag", tag_values)

    filtered = [
        paper for paper in papers
        if _matches(paper, title_query, selected_topic, selected_year, selected_tag, tags)
    ]
    paper_table(filtered)
    if not filtered:
        st.info("No papers match the current filters.")
        return

    selected = st.selectbox("Select paper", filtered, format_func=lambda paper: f"{paper['id']} - {paper['title']}")
    actions = st.columns(3)
    if actions[0].button("View Detail"):
        select_paper(selected["id"])
        st.success(f"Selected paper {selected['id']}. Open Paper Detail.")
    if actions[1].button("Search in Paper"):
        select_paper(selected["id"])
        st.session_state["last_search_query"] = selected.get("title") or ""
        st.success("Selected paper for Search.")
    if actions[2].button("Ask Paper"):
        select_paper(selected["id"])
        st.success("Selected paper for Ask Papers.")


def _matches(paper: dict, title: str, topic: str, year: str, tag: str, tags: list[dict]) -> bool:
    if title and title.lower() not in str(paper.get("title") or "").lower():
        return False
    if topic and paper.get("primary_topic") != topic:
        return False
    if year and str(paper.get("year")) != year:
        return False
    if tag and not any(str(item.get("paper_id")) == str(paper.get("id")) and item.get("tag") == tag for item in tags):
        return False
    return True

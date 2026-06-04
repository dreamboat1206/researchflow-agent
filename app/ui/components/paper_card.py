from __future__ import annotations

import streamlit as st


def paper_card(paper: dict) -> None:
    with st.container(border=True):
        st.subheader(paper.get("title") or f"Paper {paper.get('id')}")
        st.caption(
            f"paper_id: {paper.get('id')} | "
            f"topic: {paper.get('primary_topic') or '-'} | "
            f"year: {paper.get('year') or '-'}"
        )
        st.write(f"organized_path: {paper.get('organized_path') or '-'}")

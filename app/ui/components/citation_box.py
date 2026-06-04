from __future__ import annotations

import streamlit as st


def citation_box(citations: list[dict]) -> None:
    if not citations:
        st.info("No citations returned.")
        return
    for index, citation in enumerate(citations, start=1):
        st.caption(
            f"[{index}] {citation.get('title') or 'Unknown paper'} | "
            f"page: {citation.get('page') or '-'} | "
            f"chunk_id: {citation.get('chunk_id') or citation.get('figure_id') or '-'}"
        )

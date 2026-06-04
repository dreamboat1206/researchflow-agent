from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient


def render(client: ResearchFlowApiClient) -> None:
    st.header("System / Debug")
    if st.button("Check API"):
        st.json(client.health().__dict__)
    if st.button("Check Qdrant"):
        st.json(client.qdrant_status().__dict__)
    if st.button("Show Config"):
        st.json(client.system_config().__dict__)
    if st.button("Refresh Stats"):
        st.json(client.stats().__dict__)

    with st.expander("Advanced"):
        st.caption("Dangerous maintenance actions are not implemented in the UI.")
        st.button("Clear SQLite ingest data", disabled=True)
        st.button("Clear Qdrant collection", disabled=True)
        st.button("Rebuild index", disabled=True)

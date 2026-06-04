from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.ingest_result import ingest_result
from app.ui.state import select_paper


def render(client: ResearchFlowApiClient) -> None:
    st.header("Add Paper")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    auto_organize = st.checkbox("Auto organize after ingest", value=True)
    organize_mode = st.selectbox("Organize mode", ["copy", "move"])
    if organize_mode == "move":
        st.warning("Move mode will move the original PDF file. Use copy unless you are sure.")

    if st.button("Ingest Paper", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
            return
        with st.spinner("Parsing, chunking, embedding, indexing, and organizing..."):
            result = client.ingest_paper(uploaded_file, auto_organize=auto_organize, organize_mode=organize_mode)
        if not result.ok:
            st.error(result.error or "Ingest failed.")
            return
        paper_id = result.data.get("paper_id")
        if paper_id:
            select_paper(paper_id)
        ingest_result(result.data)

from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.citation_box import citation_box


def render(client: ResearchFlowApiClient) -> None:
    st.header("Ask Papers")
    st.caption("Ask returns a natural-language answer with citations.")
    question = st.text_area("Question", value=st.session_state.get("last_qa_question", ""))
    top_k = st.slider("Context chunks", 1, 20, 5)
    scope = st.selectbox("Scope", ["All papers", "Selected paper", "Selected collection"])
    show_chunks = st.checkbox("Show retrieved chunks", value=True)
    if scope == "Selected paper" and not st.session_state.get("selected_paper_id"):
        st.warning("Select a paper from Library first.")

    if st.button("Ask", type="primary"):
        result = client.ask(question, top_k=top_k, show_chunks=show_chunks)
        if not result.ok:
            st.error(result.error or "QA failed.")
            return
        st.session_state["last_qa_question"] = question
        st.session_state["last_qa_result"] = result.data

    qa_result = st.session_state.get("last_qa_result")
    if qa_result:
        answer = qa_result.get("answer") or ""
        if "未找到足够依据" in answer:
            st.warning(answer)
        else:
            st.subheader("Answer")
            st.write(answer)
        st.subheader("Citations")
        citation_box(qa_result.get("citations") or [])
        if show_chunks and qa_result.get("retrieved_chunks"):
            st.subheader("Retrieved Evidence")
            st.json(qa_result["retrieved_chunks"])

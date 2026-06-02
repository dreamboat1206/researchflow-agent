import os

import httpx
import streamlit as st


def _api_base_url() -> str:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = os.environ.get("API_PORT", "8000")
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{port}"


def _format_score(score: object) -> str:
    if isinstance(score, int | float):
        return f"{score:.4f}"
    return "-"


st.set_page_config(
    page_title="ResearchFlow-Agent",
    layout="wide",
)

st.title("ResearchFlow-Agent")
st.caption("A minimal workspace for research document workflows.")

api_base_url = _api_base_url()

home_tab, search_tab, qa_tab = st.tabs(["Overview", "Paper Search", "Paper QA"])

with home_tab:
    st.write("Project scaffold is ready. Use the API health check to verify the backend.")
    if st.button("Check local API path"):
        st.code(f"GET {api_base_url}/health")

with search_tab:
    with st.form("paper-search-form"):
        query = st.text_input("Search text", placeholder="Enter a paper topic, method, or keyword")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search")

    if submitted:
        if not query.strip():
            st.warning("Please enter search text.")
        else:
            try:
                response = httpx.post(
                    f"{api_base_url}/search/text",
                    json={"query": query, "top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
                results = response.json()["results"]
            except httpx.HTTPError as exc:
                st.error(f"Search request failed: {exc}")
            else:
                if not results:
                    st.info("No relevant chunks found.")
                for result in results:
                    title = result.get("title") or f"Paper {result.get('paper_id')}"
                    page = result.get("page") or "-"
                    with st.container(border=True):
                        st.subheader(title)
                        st.caption(f"page: {page} | score: {_format_score(result.get('score'))}")
                        st.write(result.get("chunk_text") or "")

with qa_tab:
    with st.form("paper-qa-form"):
        question = st.text_input("Question", placeholder="Ask a question about ingested papers")
        qa_top_k = st.slider("Context chunks", min_value=1, max_value=20, value=5)
        qa_submitted = st.form_submit_button("Ask")

    if qa_submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                response = httpx.post(
                    f"{api_base_url}/qa/paper",
                    json={"question": question, "top_k": qa_top_k},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"QA request failed: {exc}")
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Citations")
                citations = result.get("citations", [])
                if not citations:
                    st.info("No citations returned.")
                for citation in citations:
                    title = citation.get("title") or "Unknown paper"
                    page = citation.get("page") or "-"
                    chunk_id = citation.get("chunk_id") or "-"
                    st.caption(f"{title} | page: {page} | chunk_id: {chunk_id}")

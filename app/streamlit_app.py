import os

import httpx
import streamlit as st


def _api_base_url() -> str:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = os.environ.get("API_PORT", "8000")
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{port}"


st.set_page_config(
    page_title="ResearchFlow-Agent",
    layout="wide",
)

st.title("ResearchFlow-Agent")
st.caption("A minimal workspace for research document workflows.")

api_base_url = _api_base_url()

home_tab, search_tab = st.tabs(["概览", "论文搜索"])

with home_tab:
    st.write("Project scaffold is ready. Use the API health check to verify the backend.")
    if st.button("Check local API path"):
        st.code(f"GET {api_base_url}/health")

with search_tab:
    with st.form("paper-search-form"):
        query = st.text_input("搜索文本", placeholder="输入论文主题、方法或关键词")
        top_k = st.slider("结果数量", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("搜索")

    if submitted:
        if not query.strip():
            st.warning("请输入搜索文本。")
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
                st.error(f"搜索请求失败：{exc}")
            else:
                if not results:
                    st.info("没有找到相关 chunk。")
                for result in results:
                    title = result.get("title") or f"Paper {result.get('paper_id')}"
                    page = result.get("page") or "-"
                    score = result.get("score")
                    score_text = f"{score:.4f}" if isinstance(score, int | float) else "-"
                    with st.container(border=True):
                        st.subheader(title)
                        st.caption(f"page: {page} | score: {score_text}")
                        st.write(result.get("chunk_text") or "")

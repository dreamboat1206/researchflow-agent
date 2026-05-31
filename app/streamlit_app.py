import streamlit as st


st.set_page_config(
    page_title="ResearchFlow-Agent",
    layout="wide",
)

st.title("ResearchFlow-Agent")
st.caption("A minimal workspace for research document workflows.")

st.write("Project scaffold is ready. Use the API health check to verify the backend.")

if st.button("Check local API path"):
    st.code("GET /health")

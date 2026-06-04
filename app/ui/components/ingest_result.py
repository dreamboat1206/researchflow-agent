from __future__ import annotations

import streamlit as st


def ingest_result(result: dict) -> None:
    st.subheader("Ingest Result")
    checks = {
        "Parsed": bool(result.get("paper_id")),
        "Chunked": bool(result.get("chunks")),
        "Embedded": bool(result.get("vectors")),
        "Indexed in Qdrant": bool(result.get("collection")),
        "Organized": bool(result.get("organization")),
    }
    for label, ok in checks.items():
        st.caption(f"{'[ok]' if ok else '...'} {label}")
    if result.get("organization_error"):
        st.warning(f"Partial success: {result['organization_error']}")
    st.json(result)

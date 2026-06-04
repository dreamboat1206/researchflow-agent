from __future__ import annotations

import streamlit as st


def chunk_viewer(chunk: dict, rank: int | None = None) -> None:
    title = chunk.get("title") or f"Paper {chunk.get('paper_id') or '-'}"
    prefix = f"{rank}. " if rank is not None else ""
    with st.container(border=True):
        st.subheader(f"{prefix}{title}")
        st.caption(
            f"paper_id: {chunk.get('paper_id') or '-'} | "
            f"page: {chunk.get('page') or chunk.get('page_start') or '-'} | "
            f"chunk_id: {chunk.get('chunk_id') or '-'} | "
            f"score: {_format_score(chunk.get('score'))}"
        )
        with st.expander("Chunk text", expanded=rank == 1):
            st.write(chunk.get("chunk_text") or chunk.get("text") or "")


def _format_score(score: object) -> str:
    if isinstance(score, int | float):
        return f"{score:.4f}"
    return "-"

from __future__ import annotations

import streamlit as st


def metric_card(label: str, value: object, caption: str | None = None) -> None:
    caption_html = f'<p class="rf-muted">{caption}</p>' if caption else ""
    st.markdown(
        f"""
        <div class="rf-card">
            <div class="rf-muted">{label}</div>
            <h2>{value}</h2>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

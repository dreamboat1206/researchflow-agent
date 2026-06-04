from __future__ import annotations

import streamlit as st


def feature_card(title: str, body: str, badge: str | None = None) -> None:
    badge_html = f'<span class="rf-pill">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="rf-card">
            {badge_html}
            <h3>{title}</h3>
            <p class="rf-muted">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

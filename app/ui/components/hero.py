from __future__ import annotations

import streamlit as st


def hero(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f'<span class="rf-pill">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="rf-hero">
            {badge_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

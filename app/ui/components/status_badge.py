from __future__ import annotations

import streamlit as st


def status_badge(label: str, status: str = "unknown") -> None:
    icon = {
        "success": "[ok]",
        "warning": "!",
        "error": "x",
        "pending": "...",
        "unknown": "?",
    }.get(status, "?")
    st.caption(f"{icon} {label}")

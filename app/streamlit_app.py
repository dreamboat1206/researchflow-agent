from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.pages import add_paper, ask, dashboard, library, organizer, paper_detail, paper_reader, search, system_debug
from app.ui.state import init_state
from app.ui.styles import apply_theme


PageRenderer = Callable[[ResearchFlowApiClient], None]


PAGES: dict[str, PageRenderer] = {
    "Dashboard": dashboard.render,
    "Add Paper": add_paper.render,
    "Library": library.render,
    "Search": search.render,
    "Ask Papers": ask.render,
    "Paper Detail": paper_detail.render,
    "Paper Reader": paper_reader.render,
    "Organizer": organizer.render,
    "System / Debug": system_debug.render,
}


def main() -> None:
    st.set_page_config(page_title="ResearchFlow-Agent", layout="wide")
    apply_theme()
    init_state()

    st.sidebar.title("ResearchFlow")
    api_url = st.sidebar.text_input("API URL", value=st.session_state["api_url"])
    st.session_state["api_url"] = api_url.rstrip("/")
    page_name = st.sidebar.radio("Navigation", list(PAGES.keys()))

    client = ResearchFlowApiClient(st.session_state["api_url"])
    st.title(page_name)
    PAGES[page_name](client)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd
import streamlit as st


def paper_table(papers: list[dict]) -> None:
    rows = []
    for paper in papers:
        rows.append(
            {
                "ID": paper.get("id"),
                "Title": paper.get("title"),
                "Topic": paper.get("primary_topic") or "-",
                "Year": paper.get("year") or "-",
                "Organized": "Yes" if paper.get("organized_path") else "Pending",
                "Organized Path": paper.get("organized_path") or "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

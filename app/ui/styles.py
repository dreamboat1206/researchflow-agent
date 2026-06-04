from __future__ import annotations

import streamlit as st


PRIMARY_PURPLE = "#4B267A"
DEEP_PURPLE = "#2E174D"
SOFT_PURPLE_BACKGROUND = "#F5F0FF"
CARD_BACKGROUND = "#FFFFFF"
MINT_ACCENT = "#9BE7C0"
YELLOW_CTA = "#FFC857"
TEXT_PRIMARY = "#221A2F"
TEXT_MUTED = "#6B6178"
BORDER = "#E4DDF2"
DANGER = "#B42318"
WARNING = "#B7791F"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --rf-primary: {PRIMARY_PURPLE};
            --rf-deep: {DEEP_PURPLE};
            --rf-bg: {SOFT_PURPLE_BACKGROUND};
            --rf-card: {CARD_BACKGROUND};
            --rf-mint: {MINT_ACCENT};
            --rf-cta: {YELLOW_CTA};
            --rf-text: {TEXT_PRIMARY};
            --rf-muted: {TEXT_MUTED};
            --rf-border: {BORDER};
            --rf-danger: {DANGER};
            --rf-warning: {WARNING};
        }}

        .stApp {{
            background: var(--rf-bg);
            color: var(--rf-text);
        }}

        section[data-testid="stSidebar"] {{
            background: var(--rf-deep);
        }}

        section[data-testid="stSidebar"] * {{
            color: #FFFFFF;
        }}

        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stForm"] {{
            background: var(--rf-card);
            border: 1px solid var(--rf-border);
            border-radius: 16px;
            padding: 0.75rem;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            background: var(--rf-cta);
            border: 1px solid #E8B43E;
            border-radius: 999px;
            color: var(--rf-deep);
            font-weight: 700;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            border-color: var(--rf-primary);
            color: var(--rf-deep);
        }}

        .rf-card {{
            background: var(--rf-card);
            border: 1px solid var(--rf-border);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(46, 23, 77, 0.08);
            padding: 1.1rem;
            height: 100%;
        }}

        .rf-hero {{
            background: linear-gradient(135deg, #FFFFFF 0%, #EEE2FF 100%);
            border: 1px solid var(--rf-border);
            border-radius: 22px;
            padding: 2rem;
            margin-bottom: 1.25rem;
        }}

        .rf-hero h1 {{
            color: var(--rf-deep);
            font-size: 2.2rem;
            line-height: 1.12;
            margin: 0 0 0.7rem 0;
        }}

        .rf-hero p,
        .rf-muted {{
            color: var(--rf-muted);
        }}

        .rf-pill {{
            background: #EFE6FF;
            border: 1px solid var(--rf-border);
            border-radius: 999px;
            color: var(--rf-primary);
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            padding: 0.25rem 0.7rem;
        }}

        .rf-pill-warning {{
            background: #FFF3CD;
            border-color: #FFE08A;
            color: var(--rf-warning);
        }}

        .rf-pill-danger {{
            background: #FEE4E2;
            border-color: #FDA29B;
            color: var(--rf-danger);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

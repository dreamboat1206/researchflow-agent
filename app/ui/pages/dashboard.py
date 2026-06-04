from __future__ import annotations

import streamlit as st

from app.ui.api_client import ResearchFlowApiClient
from app.ui.components.feature_card import feature_card
from app.ui.components.hero import hero
from app.ui.components.metric_card import metric_card
from app.ui.components.paper_card import paper_card
from app.ui.components.status_badge import status_badge


def render(client: ResearchFlowApiClient) -> None:
    hero(
        "Summarize, search and organize your papers",
        "Upload PDFs, build a searchable research library, ask grounded questions, and keep every paper organized.",
        badge="ResearchFlow Agent",
    )

    health = client.health()
    stats = client.stats()
    qdrant = client.qdrant_status()
    config = client.system_config()

    feature_cols = st.columns(3)
    with feature_cols[0]:
        feature_card("Add & Organize", "Ingest PDFs, extract metadata, index chunks, and keep files in a clean library.")
    with feature_cols[1]:
        feature_card("Search Evidence", "Retrieve text chunks and figures with scores, pages, and source context.")
    with feature_cols[2]:
        feature_card("Ask with Citations", "Generate grounded answers that keep paper titles, pages, and chunk ids visible.")

    st.subheader("System Snapshot")
    cols = st.columns(4)
    data = stats.data if stats.ok else {}
    with cols[0]:
        metric_card("Total Papers", data.get("total_papers", "Unknown"), "SQLite paper records")
    with cols[1]:
        metric_card("Total Chunks", data.get("total_chunks", "Unknown"), "Indexed text evidence")
    with cols[2]:
        metric_card("Organized Papers", data.get("organized_papers", "Unknown"), "Files with organized paths")
    with cols[3]:
        metric_card("Qdrant", "Connected" if qdrant.ok and qdrant.data.get("connected") else "Unknown", "Vector service")

    status_badge("API connected", "success" if health.ok else "error")
    status_badge("SQLite connected", "success" if data.get("sqlite_connected") else "unknown")
    status_badge("Qdrant connected", "success" if qdrant.ok and qdrant.data.get("connected") else "warning")

    if config.ok:
        models = config.data.get("models", {})
        qdrant_config = config.data.get("qdrant", {})
        st.caption(f"Embedding model: {models.get('embedding_model') or '-'}")
        st.caption(f"Collection: {qdrant_config.get('collection_name') or '-'}")
    else:
        st.caption("Config is unavailable. Start the API service to show model and collection settings.")

    st.subheader("Recent Papers")
    recent_papers = data.get("recent_papers", [])
    if not recent_papers:
        st.info("No recent papers yet, or the stats API is unavailable.")
    for paper in recent_papers:
        paper_card(paper)

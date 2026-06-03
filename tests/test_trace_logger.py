from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from graph.figure_search_graph import invoke_figure_search
from observability.trace_logger import TRACE_EVENT_TYPE, TraceLogger, list_recent_traces
from storage.sqlite_store import get_db_path, init_db


class FakeRouterAgent:
    def route_query(self, user_input: str) -> dict[str, object]:
        return {"task_type": "figure_search", "query": user_input, "need_multimodal": True}


class FakeFigureRetrievalAgent:
    def search_figures(self, query: str, top_k: int, mode: str = "fusion") -> list[dict[str, object]]:
        return [
            {
                "figure_id": "1_fig_2_1",
                "paper_id": 1,
                "page": 2,
                "image_path": "data/figures/1/1_fig_2_1.png",
                "caption": f"Figure for {query}",
                "score": 0.9,
            }
        ][:top_k]


class FailingFigureRetrievalAgent:
    def search_figures(self, query: str, top_k: int, mode: str = "fusion") -> list[dict[str, object]]:
        raise RuntimeError("retrieval failed")


def test_trace_logger_writes_and_lists_sqlite_trace() -> None:
    config_path = _write_config()
    logger = TraceLogger(config_path=config_path)

    logger.log_trace(
        trace_id="trace-1",
        task_type="paper_qa",
        query="What is ZoomDet?",
        retrieved_items=[{"type": "chunk", "chunk_id": "1-p2-c1"}],
        final_answer="Answer [1]",
        latency_ms=12.5,
        success=True,
    )

    traces = list_recent_traces(limit=5, config_path=config_path)

    assert traces[0]["trace_id"] == "trace-1"
    assert traces[0]["task_type"] == "paper_qa"
    assert traces[0]["query"] == "What is ZoomDet?"
    assert traces[0]["retrieved_items"] == [{"type": "chunk", "chunk_id": "1-p2-c1"}]
    assert traces[0]["latency_ms"] == 12.5
    assert traces[0]["success"] is True

    with sqlite3.connect(get_db_path(config_path)) as connection:
        row = connection.execute("SELECT run_id, event_type FROM traces").fetchone()
    assert row == ("trace-1", TRACE_EVENT_TYPE)


def test_trace_logger_records_exception() -> None:
    config_path = _write_config()
    logger = TraceLogger(config_path=config_path)
    trace_id, started_at = logger.start_trace()

    logger.log_exception(
        trace_id=trace_id,
        started_at=started_at,
        task_type="figure_search",
        query="find figure",
        error=RuntimeError("boom"),
    )

    trace = list_recent_traces(limit=1, config_path=config_path)[0]
    assert trace["success"] is False
    assert trace["error_message"] == "RuntimeError: boom"
    assert trace["retrieved_items"] == []


def test_graph_invoke_records_success_trace() -> None:
    config_path = _write_config()

    result = invoke_figure_search(
        "architecture",
        top_k=1,
        router_agent=FakeRouterAgent(),
        figure_retrieval_agent=FakeFigureRetrievalAgent(),
        config_path=config_path,
    )

    trace = list_recent_traces(limit=1, config_path=config_path)[0]
    assert trace["trace_id"] == result["run_id"]
    assert trace["task_type"] == "figure_search"
    assert trace["success"] is True
    assert trace["retrieved_items"] == [
        {
            "type": "figure",
            "paper_id": 1,
            "figure_id": "1_fig_2_1",
            "page": 2,
            "score": 0.9,
        }
    ]


def test_graph_invoke_records_error_trace() -> None:
    config_path = _write_config()

    try:
        invoke_figure_search(
            "architecture",
            top_k=1,
            router_agent=FakeRouterAgent(),
            figure_retrieval_agent=FailingFigureRetrievalAgent(),
            config_path=config_path,
        )
    except RuntimeError:
        pass

    trace = list_recent_traces(limit=1, config_path=config_path)[0]
    assert trace["success"] is False
    assert trace["task_type"] == "figure_search"
    assert "retrieval failed" in trace["error_message"]


def _write_config() -> Path:
    test_dir = Path("data/test_trace_logger") / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: researchflow-test.db\n",
        encoding="utf-8",
    )
    init_db(config_path)
    return config_path

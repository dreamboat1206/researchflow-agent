from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from storage.sqlite_store import DEFAULT_CONFIG_PATH, connect, get_db_path, init_db


TRACE_EVENT_TYPE = "agent_trace"


class TraceLogger:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path

    def start_trace(self) -> tuple[str, float]:
        return uuid.uuid4().hex, time.perf_counter()

    def log_graph_result(
        self,
        trace_id: str,
        started_at: float,
        task_type: str,
        query: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        error_message = state.get("error")
        return self.log_trace(
            trace_id=trace_id,
            task_type=task_type,
            query=query,
            retrieved_items=_retrieved_items_from_state(state),
            final_answer=state.get("final_answer") or state.get("answer"),
            latency_ms=_latency_ms(started_at),
            success=not bool(error_message),
            error_message=str(error_message) if error_message else None,
            metadata={
                "citations": state.get("citations") or [],
                "evaluations": state.get("evaluations") or [],
            },
        )

    def log_exception(
        self,
        trace_id: str,
        started_at: float,
        task_type: str,
        query: str,
        error: Exception,
    ) -> dict[str, Any]:
        return self.log_trace(
            trace_id=trace_id,
            task_type=task_type,
            query=query,
            retrieved_items=[],
            final_answer=None,
            latency_ms=_latency_ms(started_at),
            success=False,
            error_message=f"{type(error).__name__}: {error}",
        )

    def log_trace(
        self,
        trace_id: str,
        task_type: str,
        query: str,
        retrieved_items: list[dict[str, Any]],
        final_answer: str | None,
        latency_ms: float,
        success: bool,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "trace_id": trace_id,
            "task_type": task_type,
            "query": query,
            "retrieved_items": retrieved_items,
            "final_answer": final_answer,
            "latency_ms": latency_ms,
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {},
        }
        _write_trace_payload(trace_id, payload, self.config_path)
        _sync_langfuse_if_configured(payload)
        return payload

    def list_recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        return list_recent_traces(limit=limit, config_path=self.config_path)


def list_recent_traces(
    limit: int = 20,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    init_db(config_path)
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, run_id, event_type, payload, created_at
            FROM traces
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (TRACE_EVENT_TYPE, limit),
        ).fetchall()

    traces: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload"] or "{}")
        payload.setdefault("trace_id", row["run_id"])
        payload["created_at"] = row["created_at"]
        traces.append(payload)
    return traces


def _write_trace_payload(
    trace_id: str,
    payload: dict[str, Any],
    config_path: str | Path,
) -> None:
    init_db(config_path)
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO traces (run_id, event_type, payload)
            VALUES (?, ?, ?)
            """,
            (trace_id, TRACE_EVENT_TYPE, json.dumps(payload, ensure_ascii=False)),
        )


def _retrieved_items_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = state.get("retrieved_chunks") or state.get("context_chunks") or []
    figures = state.get("retrieved_figures") or state.get("context_figures") or []
    items: list[dict[str, Any]] = []
    for chunk in chunks:
        items.append(
            {
                "type": "chunk",
                "paper_id": chunk.get("paper_id"),
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "score": chunk.get("score"),
            }
        )
    for figure in figures:
        items.append(
            {
                "type": "figure",
                "paper_id": figure.get("paper_id"),
                "figure_id": figure.get("figure_id"),
                "page": figure.get("page"),
                "score": figure.get("final_score") or figure.get("score"),
            }
        )
    return items


def _latency_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _sync_langfuse_if_configured(payload: dict[str, Any]) -> None:
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return
    try:
        from langfuse import Langfuse
    except ImportError:
        return

    client = Langfuse()
    trace = client.trace(
        id=str(payload["trace_id"]),
        name=str(payload["task_type"]),
        input={"query": payload["query"]},
        output={"final_answer": payload.get("final_answer")},
        metadata=payload,
    )
    trace.score(
        name="success",
        value=1.0 if payload.get("success") else 0.0,
    )

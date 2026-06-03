from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.figure_retrieval_agent import FigureRetrievalAgent
from agents.router_agent import RouterAgent
from graph.state import FigureSearchMode, ResearchState, create_initial_state
from observability.trace_logger import TraceLogger
from storage.sqlite_store import DEFAULT_CONFIG_PATH


def router_node(
    state: ResearchState,
    router_agent: RouterAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    query = str(next_state.get("figure_query") or next_state.get("user_query") or "").strip()
    router = router_agent or RouterAgent()
    route = router.route_query(query)
    next_state["workflow"] = "figure_search"
    next_state["user_query"] = route["query"]
    next_state["figure_query"] = route["query"]
    next_state["task_type"] = route["task_type"]
    next_state["need_multimodal"] = True
    if "figure_search_mode" not in next_state:
        next_state["figure_search_mode"] = "fusion"
    _append_trace(next_state, "router_node", {"task_type": route["task_type"]})
    return next_state


def figure_retrieval_node(
    state: ResearchState,
    figure_retrieval_agent: FigureRetrievalAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    if next_state.get("error"):
        return next_state

    task_type = next_state.get("task_type")
    if task_type not in (None, "figure_search", "figure_qa", "unknown", "paper_qa"):
        next_state["error"] = f"Query routed to {task_type}, not figure_search."
        _append_trace(next_state, "figure_retrieval_node", {"skipped": True})
        return next_state

    query = str(next_state.get("figure_query") or next_state.get("user_query") or "").strip()
    top_k = int(next_state.get("top_k") or 5)
    mode = str(next_state.get("figure_search_mode") or "fusion")
    agent = figure_retrieval_agent or FigureRetrievalAgent()
    figures = agent.search_figures(query, top_k=top_k, mode=mode)
    next_state["retrieved_figures"] = figures
    next_state["context_figures"] = figures
    _append_trace(
        next_state,
        "figure_retrieval_node",
        {"figure_count": len(figures), "mode": mode},
    )
    return next_state


def format_results_node(state: ResearchState) -> ResearchState:
    next_state = _copy_state(state)
    figures = [_format_figure(figure) for figure in list(next_state.get("retrieved_figures") or [])]
    next_state["retrieved_figures"] = figures
    next_state["context_figures"] = figures
    _append_trace(next_state, "format_results_node", {"figure_count": len(figures)})
    return next_state


def build_figure_search_graph(
    router_agent: RouterAgent | None = None,
    figure_retrieval_agent: FigureRetrievalAgent | None = None,
) -> Any:
    nodes = [
        lambda state: router_node(state, router_agent=router_agent),
        lambda state: figure_retrieval_node(
            state,
            figure_retrieval_agent=figure_retrieval_agent,
        ),
        format_results_node,
    ]
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialGraph(nodes)

    graph = StateGraph(ResearchState)
    graph.add_node("router_node", nodes[0])
    graph.add_node("figure_retrieval_node", nodes[1])
    graph.add_node("format_results_node", nodes[2])
    graph.set_entry_point("router_node")
    graph.add_edge("router_node", "figure_retrieval_node")
    graph.add_edge("figure_retrieval_node", "format_results_node")
    graph.add_edge("format_results_node", END)
    return graph.compile()


def invoke_figure_search(
    query: str,
    top_k: int = 5,
    mode: FigureSearchMode = "fusion",
    router_agent: RouterAgent | None = None,
    figure_retrieval_agent: FigureRetrievalAgent | None = None,
    trace_logger: TraceLogger | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    if router_agent is None:
        router_agent = RouterAgent(config_path=config_path)
    if figure_retrieval_agent is None:
        figure_retrieval_agent = FigureRetrievalAgent(config_path=config_path)

    graph = build_figure_search_graph(
        router_agent=router_agent,
        figure_retrieval_agent=figure_retrieval_agent,
    )
    initial_state = create_initial_state("figure_search", user_query=query, top_k=top_k)
    initial_state["figure_query"] = query
    initial_state["figure_search_mode"] = mode
    logger = trace_logger or TraceLogger(config_path=config_path)
    trace_id, started_at = logger.start_trace()
    initial_state["run_id"] = trace_id
    try:
        result = graph.invoke(initial_state)
        result["run_id"] = trace_id
        logger.log_graph_result(
            trace_id=trace_id,
            started_at=started_at,
            task_type="figure_search",
            query=query,
            state=result,
        )
        return result
    except Exception as error:
        logger.log_exception(
            trace_id=trace_id,
            started_at=started_at,
            task_type="figure_search",
            query=query,
            error=error,
        )
        raise


class _SequentialGraph:
    def __init__(self, nodes: list[Callable[[ResearchState], ResearchState]]):
        self.nodes = nodes

    def invoke(self, state: ResearchState) -> ResearchState:
        current_state = state
        for node in self.nodes:
            current_state = node(current_state)
        return current_state


def _format_figure(figure: dict[str, Any]) -> dict[str, Any]:
    return {
        "figure_id": figure.get("figure_id"),
        "paper_id": figure.get("paper_id"),
        "page": figure.get("page"),
        "image_path": figure.get("image_path"),
        "caption": figure.get("caption"),
        "nearby_text": figure.get("nearby_text"),
        "figure_type": figure.get("figure_type"),
        "score": figure.get("score"),
        "final_score": figure.get("final_score"),
        "score_breakdown": figure.get("score_breakdown"),
    }


def _copy_state(state: ResearchState) -> ResearchState:
    copied: ResearchState = dict(state)
    for key in ("messages", "trace", "retrieved_chunks", "retrieved_figures", "citations", "evaluations"):
        if key in copied:
            copied[key] = list(copied[key])  # type: ignore[literal-required]
    return copied


def _append_trace(state: ResearchState, node_name: str, payload: dict[str, Any]) -> None:
    trace = list(state.get("trace") or [])
    trace.append({"node": node_name, "payload": payload})
    state["trace"] = trace

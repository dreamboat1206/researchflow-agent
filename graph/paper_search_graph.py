from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.retrieval_agent import RetrievalAgent
from agents.router_agent import RouterAgent
from graph.state import ResearchState, create_initial_state
from observability.trace_logger import TraceLogger
from storage.sqlite_store import DEFAULT_CONFIG_PATH


def router_node(state: ResearchState, router_agent: RouterAgent | None = None) -> ResearchState:
    next_state = _copy_state(state)
    query = str(next_state.get("user_query") or "").strip()
    route = (router_agent or RouterAgent()).route_query(query)
    next_state["workflow"] = "paper_search"
    next_state["user_query"] = route["query"]
    next_state["task_type"] = "paper_search"
    next_state["need_multimodal"] = False
    _append_trace(next_state, "router_node", {"task_type": route["task_type"]})
    return next_state


def text_retrieval_node(
    state: ResearchState,
    retrieval_agent: RetrievalAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    query = str(next_state.get("user_query") or "").strip()
    top_k = int(next_state.get("top_k") or 5)
    chunks = (retrieval_agent or RetrievalAgent()).search_papers(query, top_k=top_k)
    next_state["retrieved_chunks"] = chunks
    next_state["context_chunks"] = chunks
    _append_trace(next_state, "text_retrieval_node", {"chunk_count": len(chunks)})
    return next_state


def format_results_node(state: ResearchState) -> ResearchState:
    next_state = _copy_state(state)
    chunks = [_format_chunk(chunk) for chunk in list(next_state.get("retrieved_chunks") or [])]
    next_state["retrieved_chunks"] = chunks
    next_state["context_chunks"] = chunks
    _append_trace(next_state, "format_results_node", {"chunk_count": len(chunks)})
    return next_state


def build_paper_search_graph(
    router_agent: RouterAgent | None = None,
    retrieval_agent: RetrievalAgent | None = None,
) -> Any:
    nodes = [
        lambda state: router_node(state, router_agent=router_agent),
        lambda state: text_retrieval_node(state, retrieval_agent=retrieval_agent),
        format_results_node,
    ]
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialGraph(nodes)

    graph = StateGraph(ResearchState)
    graph.add_node("router_node", nodes[0])
    graph.add_node("text_retrieval_node", nodes[1])
    graph.add_node("format_results_node", nodes[2])
    graph.set_entry_point("router_node")
    graph.add_edge("router_node", "text_retrieval_node")
    graph.add_edge("text_retrieval_node", "format_results_node")
    graph.add_edge("format_results_node", END)
    return graph.compile()


def invoke_paper_search(
    query: str,
    top_k: int = 5,
    router_agent: RouterAgent | None = None,
    retrieval_agent: RetrievalAgent | None = None,
    trace_logger: TraceLogger | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    if router_agent is None:
        router_agent = RouterAgent(config_path=config_path)
    if retrieval_agent is None:
        retrieval_agent = RetrievalAgent(config_path=config_path)
    graph = build_paper_search_graph(router_agent=router_agent, retrieval_agent=retrieval_agent)
    initial_state = create_initial_state("paper_search", user_query=query, top_k=top_k)
    logger = trace_logger or TraceLogger(config_path=config_path)
    trace_id, started_at = logger.start_trace()
    initial_state["run_id"] = trace_id
    try:
        result = graph.invoke(initial_state)
        result["run_id"] = trace_id
        logger.log_graph_result(
            trace_id=trace_id,
            started_at=started_at,
            task_type="paper_search",
            query=query,
            state=result,
        )
        return result
    except Exception as error:
        logger.log_exception(
            trace_id=trace_id,
            started_at=started_at,
            task_type="paper_search",
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


def _format_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": chunk.get("title"),
        "paper_id": chunk.get("paper_id"),
        "chunk_id": chunk.get("chunk_id"),
        "page": chunk.get("page"),
        "chunk_text": chunk.get("chunk_text"),
        "score": chunk.get("score"),
        "metadata": chunk.get("metadata"),
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

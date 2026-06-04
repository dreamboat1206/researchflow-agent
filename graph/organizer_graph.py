from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.organizer_agent import OrganizerAgent
from graph.state import ResearchState, create_initial_state
from observability.trace_logger import TraceLogger
from storage.sqlite_store import DEFAULT_CONFIG_PATH


def list_papers_node(
    state: ResearchState,
    organizer_agent: OrganizerAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    agent = organizer_agent or OrganizerAgent()
    limit = next_state.get("limit")
    next_state["papers"] = agent.sqlite_store.list_papers(limit=limit, config_path=agent.config_path)
    _append_trace(next_state, "list_papers_node", {"paper_count": len(next_state["papers"])})
    return next_state


def organize_paper_node(
    state: ResearchState,
    organizer_agent: OrganizerAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    agent = organizer_agent or OrganizerAgent()
    paper_id = str(next_state.get("paper_id") or "").strip()
    result = agent.organize_paper(
        paper_id,
        dry_run=bool(next_state.get("dry_run", True)),
        mode=next_state.get("mode"),
    )
    next_state["organization_result"] = result.to_dict()
    _append_trace(next_state, "organize_paper_node", {"paper_id": paper_id})
    return next_state


def organize_papers_node(
    state: ResearchState,
    organizer_agent: OrganizerAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    agent = organizer_agent or OrganizerAgent()
    results = agent.organize_all_papers(
        dry_run=bool(next_state.get("dry_run", True)),
        mode=next_state.get("mode"),
        limit=next_state.get("limit"),
    )
    next_state["organization_results"] = [result.to_dict() for result in results]
    _append_trace(next_state, "organize_papers_node", {"result_count": len(results)})
    return next_state


def build_organizer_graph(
    action: str,
    organizer_agent: OrganizerAgent | None = None,
) -> Any:
    if action == "list":
        nodes = [lambda state: list_papers_node(state, organizer_agent=organizer_agent)]
    elif action == "single":
        nodes = [lambda state: organize_paper_node(state, organizer_agent=organizer_agent)]
    elif action == "batch":
        nodes = [lambda state: organize_papers_node(state, organizer_agent=organizer_agent)]
    else:
        raise ValueError(f"Unsupported organizer graph action: {action}")

    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialGraph(nodes)

    graph = StateGraph(ResearchState)
    graph.add_node(f"{action}_node", nodes[0])
    graph.set_entry_point(f"{action}_node")
    graph.add_edge(f"{action}_node", END)
    return graph.compile()


def invoke_list_organizer_papers(
    limit: int | None = 50,
    organizer_agent: OrganizerAgent | None = None,
    trace_logger: TraceLogger | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    if organizer_agent is None:
        organizer_agent = OrganizerAgent(config_path=config_path)
    state = create_initial_state("organizer", user_query="list organizer papers")
    state["limit"] = limit
    return _invoke("list", state, "organizer_list", organizer_agent, trace_logger, config_path)


def invoke_organize_paper(
    paper_id: str,
    dry_run: bool = True,
    mode: str | None = None,
    organizer_agent: OrganizerAgent | None = None,
    trace_logger: TraceLogger | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    if organizer_agent is None:
        organizer_agent = OrganizerAgent(config_path=config_path)
    state = create_initial_state("organizer", user_query=f"organize paper {paper_id}")
    state["paper_id"] = int(paper_id) if str(paper_id).isdigit() else paper_id  # type: ignore[typeddict-item]
    state["dry_run"] = dry_run
    state["mode"] = mode
    return _invoke("single", state, "organizer_single", organizer_agent, trace_logger, config_path)


def invoke_organize_papers(
    dry_run: bool = True,
    mode: str | None = None,
    limit: int | None = None,
    organizer_agent: OrganizerAgent | None = None,
    trace_logger: TraceLogger | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    if organizer_agent is None:
        organizer_agent = OrganizerAgent(config_path=config_path)
    state = create_initial_state("organizer", user_query="organize papers")
    state["dry_run"] = dry_run
    state["mode"] = mode
    state["limit"] = limit
    return _invoke("batch", state, "organizer_batch", organizer_agent, trace_logger, config_path)


def _invoke(
    action: str,
    state: ResearchState,
    task_type: str,
    organizer_agent: OrganizerAgent,
    trace_logger: TraceLogger | None,
    config_path: str | Path,
) -> ResearchState:
    graph = build_organizer_graph(action, organizer_agent=organizer_agent)
    logger = trace_logger or TraceLogger(config_path=config_path)
    trace_id, started_at = logger.start_trace()
    state["run_id"] = trace_id
    try:
        result = graph.invoke(state)
        result["run_id"] = trace_id
        logger.log_graph_result(
            trace_id=trace_id,
            started_at=started_at,
            task_type=task_type,
            query=str(state.get("user_query") or ""),
            state=result,
        )
        return result
    except Exception as error:
        logger.log_exception(
            trace_id=trace_id,
            started_at=started_at,
            task_type=task_type,
            query=str(state.get("user_query") or ""),
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


def _copy_state(state: ResearchState) -> ResearchState:
    copied: ResearchState = dict(state)
    for key in (
        "messages",
        "trace",
        "retrieved_chunks",
        "retrieved_figures",
        "citations",
        "evaluations",
        "papers",
        "organization_results",
    ):
        if key in copied:
            copied[key] = list(copied[key])  # type: ignore[literal-required]
    return copied


def _append_trace(state: ResearchState, node_name: str, payload: dict[str, Any]) -> None:
    trace = list(state.get("trace") or [])
    trace.append({"node": node_name, "payload": payload})
    state["trace"] = trace

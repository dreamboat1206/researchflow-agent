from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.evaluator_agent import EvaluatorAgent
from agents.figure_retrieval_agent import FigureRetrievalAgent
from agents.multimodal_qa_agent import FIGURE_NOT_FOUND_ERROR, MultimodalQAAgent
from agents.router_agent import RouterAgent
from graph.state import ResearchState, create_initial_state
from storage.sqlite_store import DEFAULT_CONFIG_PATH, get_figure_by_id


def router_node(
    state: ResearchState,
    router_agent: RouterAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    query = str(next_state.get("question") or next_state.get("user_query") or "").strip()
    router = router_agent or RouterAgent()
    route = router.route_query(query)
    next_state["workflow"] = "figure_qa"
    next_state["user_query"] = route["query"]
    next_state["question"] = route["query"]
    next_state["task_type"] = "figure_qa" if route["task_type"] in {"unknown", "paper_qa"} else route["task_type"]
    next_state["need_multimodal"] = True
    _append_trace(next_state, "router_node", {"task_type": next_state["task_type"]})
    return next_state


def figure_lookup_node(
    state: ResearchState,
    figure_retrieval_agent: FigureRetrievalAgent | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    figure_id = str(next_state.get("figure_id") or "").strip()
    if figure_id:
        figure = get_figure_by_id(figure_id, config_path=config_path)
        if figure:
            next_state["selected_figure"] = figure
            next_state["retrieved_figures"] = [figure]
            next_state["context_figures"] = [figure]
            _append_trace(next_state, "figure_lookup_node", {"source": "figure_id", "found": True})
            return next_state
        next_state["error"] = f"{FIGURE_NOT_FOUND_ERROR} figure_id={figure_id}"
        _append_trace(next_state, "figure_lookup_node", {"source": "figure_id", "found": False})
        return next_state

    query = str(next_state.get("figure_query") or next_state.get("question") or next_state.get("user_query") or "")
    top_k = int(next_state.get("top_k") or 5)
    mode = str(next_state.get("figure_search_mode") or "fusion")
    agent = figure_retrieval_agent or FigureRetrievalAgent(config_path=config_path)
    figures = agent.search_figures(query, top_k=top_k, mode=mode)
    next_state["retrieved_figures"] = figures
    next_state["context_figures"] = figures
    if figures:
        next_state["selected_figure"] = figures[0]
    else:
        next_state["error"] = FIGURE_NOT_FOUND_ERROR
    _append_trace(next_state, "figure_lookup_node", {"source": "query", "figure_count": len(figures)})
    return next_state


def answer_generation_node(
    state: ResearchState,
    multimodal_qa_agent: MultimodalQAAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    agent = multimodal_qa_agent or MultimodalQAAgent()
    question = str(next_state.get("question") or next_state.get("user_query") or "").strip()

    if next_state.get("error"):
        result = agent.answer_figure_question(question, None)
    else:
        result = agent.answer_figure_question(question, next_state.get("selected_figure"))

    next_state["answer"] = result["answer"]
    next_state["final_answer"] = result["answer"]
    if result.get("error"):
        next_state["error"] = str(result["error"])
    next_state["paper_id"] = result.get("paper_id")  # type: ignore[typeddict-item]
    if result.get("figure_id"):
        next_state["figure_id"] = result["figure_id"]  # type: ignore[typeddict-unknown-key]
    if result.get("page"):
        next_state["page"] = result["page"]  # type: ignore[typeddict-unknown-key]
    next_state["citations"] = _figure_citations(result)
    _append_trace(next_state, "answer_generation_node", {"has_error": bool(result.get("error"))})
    return next_state


def evaluation_node(
    state: ResearchState,
    evaluator_agent: EvaluatorAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    evaluator = evaluator_agent or EvaluatorAgent()
    evaluation = evaluator.evaluate_state(next_state)
    next_state["evaluations"] = list(next_state.get("evaluations") or []) + [evaluation]
    _append_trace(next_state, "evaluation_node", {"passed": evaluation["passed"]})
    return next_state


def format_output_node(state: ResearchState) -> ResearchState:
    next_state = _copy_state(state)
    selected = dict(next_state.get("selected_figure") or {})
    if selected:
        next_state["selected_figure"] = {
            "figure_id": selected.get("figure_id"),
            "paper_id": selected.get("paper_id"),
            "page": selected.get("page"),
            "image_path": selected.get("image_path"),
            "caption": selected.get("caption"),
            "nearby_text": selected.get("nearby_text"),
            "figure_type": selected.get("figure_type"),
            "score": selected.get("score"),
            "final_score": selected.get("final_score"),
            "score_breakdown": selected.get("score_breakdown"),
            "metadata": selected.get("metadata"),
        }
    _append_trace(next_state, "format_output_node", {"has_figure": bool(selected)})
    return next_state


def build_figure_qa_graph(
    router_agent: RouterAgent | None = None,
    figure_retrieval_agent: FigureRetrievalAgent | None = None,
    multimodal_qa_agent: MultimodalQAAgent | None = None,
    evaluator_agent: EvaluatorAgent | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Any:
    nodes = [
        lambda state: router_node(state, router_agent=router_agent),
        lambda state: figure_lookup_node(
            state,
            figure_retrieval_agent=figure_retrieval_agent,
            config_path=config_path,
        ),
        lambda state: answer_generation_node(state, multimodal_qa_agent=multimodal_qa_agent),
        lambda state: evaluation_node(state, evaluator_agent=evaluator_agent),
        format_output_node,
    ]
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialGraph(nodes)

    graph = StateGraph(ResearchState)
    graph.add_node("router_node", nodes[0])
    graph.add_node("figure_lookup_node", nodes[1])
    graph.add_node("answer_generation_node", nodes[2])
    graph.add_node("evaluation_node", nodes[3])
    graph.add_node("format_output_node", nodes[4])
    graph.set_entry_point("router_node")
    graph.add_edge("router_node", "figure_lookup_node")
    graph.add_edge("figure_lookup_node", "answer_generation_node")
    graph.add_edge("answer_generation_node", "evaluation_node")
    graph.add_edge("evaluation_node", "format_output_node")
    graph.add_edge("format_output_node", END)
    return graph.compile()


def invoke_figure_qa(
    question: str,
    figure_id: str | None = None,
    query: str | None = None,
    top_k: int = 5,
    router_agent: RouterAgent | None = None,
    figure_retrieval_agent: FigureRetrievalAgent | None = None,
    multimodal_qa_agent: MultimodalQAAgent | None = None,
    evaluator_agent: EvaluatorAgent | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    graph = build_figure_qa_graph(
        router_agent=router_agent,
        figure_retrieval_agent=figure_retrieval_agent,
        multimodal_qa_agent=multimodal_qa_agent,
        evaluator_agent=evaluator_agent,
        config_path=config_path,
    )
    initial_state = create_initial_state("figure_qa", user_query=question, top_k=top_k)
    initial_state["question"] = question
    if figure_id:
        initial_state["figure_id"] = figure_id  # type: ignore[typeddict-unknown-key]
    if query:
        initial_state["figure_query"] = query
    initial_state["figure_search_mode"] = "fusion"
    return graph.invoke(initial_state)


class _SequentialGraph:
    def __init__(self, nodes: list[Callable[[ResearchState], ResearchState]]):
        self.nodes = nodes

    def invoke(self, state: ResearchState) -> ResearchState:
        current_state = state
        for node in self.nodes:
            current_state = node(current_state)
        return current_state


def _figure_citations(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not result.get("figure_id"):
        return []
    return [
        {
            "paper_id": result.get("paper_id"),
            "figure_id": result.get("figure_id"),
            "page": result.get("page"),
            "evidence": result.get("caption"),
        }
    ]


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

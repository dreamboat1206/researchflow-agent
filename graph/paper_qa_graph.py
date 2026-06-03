from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.paper_qa_agent import (
    INSUFFICIENT_CONTEXT_ANSWER,
    _build_prompt,
    _citation_from_chunk,
    _ensure_citation_note,
)
from agents.retrieval_agent import RetrievalAgent
from agents.router_agent import RouterAgent
from graph.state import ResearchState, create_initial_state
from models.llm_client import LLMClient
from storage.sqlite_store import DEFAULT_CONFIG_PATH


def router_node(
    state: ResearchState,
    router_agent: RouterAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    query = str(next_state.get("question") or next_state.get("user_query") or "").strip()
    router = router_agent or RouterAgent()
    route = router.route_query(query)
    next_state["workflow"] = "paper_qa"
    next_state["user_query"] = route["query"]
    next_state["question"] = route["query"]
    next_state["task_type"] = route["task_type"]
    next_state["need_multimodal"] = route["need_multimodal"]
    _append_trace(next_state, "router_node", {"task_type": route["task_type"]})
    return next_state


def text_retrieval_node(
    state: ResearchState,
    retrieval_agent: RetrievalAgent | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    if next_state.get("error"):
        return next_state

    task_type = next_state.get("task_type")
    if task_type not in (None, "paper_qa", "unknown"):
        next_state["error"] = f"Query routed to {task_type}, not paper_qa."
        _append_trace(next_state, "text_retrieval_node", {"skipped": True})
        return next_state

    query = str(next_state.get("question") or next_state.get("user_query") or "").strip()
    top_k = int(next_state.get("top_k") or 5)
    retriever = retrieval_agent or RetrievalAgent()
    chunks = retriever.search_papers(query, top_k=top_k)
    next_state["retrieved_chunks"] = chunks
    next_state["context_chunks"] = chunks
    _append_trace(next_state, "text_retrieval_node", {"chunk_count": len(chunks)})
    return next_state


def answer_generation_node(
    state: ResearchState,
    llm_client: LLMClient | None = None,
) -> ResearchState:
    next_state = _copy_state(state)
    if next_state.get("error"):
        next_state["answer"] = INSUFFICIENT_CONTEXT_ANSWER
        next_state["final_answer"] = INSUFFICIENT_CONTEXT_ANSWER
        next_state["citations"] = []
        return next_state

    chunks = list(next_state.get("context_chunks") or next_state.get("retrieved_chunks") or [])
    if not chunks:
        next_state["answer"] = INSUFFICIENT_CONTEXT_ANSWER
        next_state["final_answer"] = INSUFFICIENT_CONTEXT_ANSWER
        next_state["citations"] = []
        _append_trace(next_state, "answer_generation_node", {"has_context": False})
        return next_state

    question = str(next_state.get("question") or next_state.get("user_query") or "").strip()
    citations = [_citation_from_chunk(chunk) for chunk in chunks]
    prompt = _build_prompt(question, chunks)
    client = llm_client or LLMClient()
    answer = _ensure_citation_note(client.generate(prompt), citations)
    next_state["prompt"] = prompt
    next_state["citations"] = citations
    next_state["answer"] = answer
    next_state["final_answer"] = answer
    _append_trace(next_state, "answer_generation_node", {"citation_count": len(citations)})
    return next_state


def evaluation_node(state: ResearchState) -> ResearchState:
    next_state = _copy_state(state)
    citations = list(next_state.get("citations") or [])
    final_answer = str(next_state.get("final_answer") or "")
    supported = bool(citations) and final_answer != INSUFFICIENT_CONTEXT_ANSWER and not next_state.get("error")
    next_state["evaluations"] = list(next_state.get("evaluations") or []) + [
        {
            "metric_name": "citation_coverage",
            "score": 1.0 if supported else 0.0,
            "passed": supported,
            "details": {
                "citation_count": len(citations),
                "has_answer": bool(final_answer),
            },
        }
    ]
    _append_trace(next_state, "evaluation_node", {"passed": supported})
    return next_state


def build_paper_qa_graph(
    router_agent: RouterAgent | None = None,
    retrieval_agent: RetrievalAgent | None = None,
    llm_client: LLMClient | None = None,
) -> Any:
    nodes = [
        lambda state: router_node(state, router_agent=router_agent),
        lambda state: text_retrieval_node(state, retrieval_agent=retrieval_agent),
        lambda state: answer_generation_node(state, llm_client=llm_client),
        evaluation_node,
    ]
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialGraph(nodes)

    graph = StateGraph(ResearchState)
    graph.add_node("router_node", nodes[0])
    graph.add_node("text_retrieval_node", nodes[1])
    graph.add_node("answer_generation_node", nodes[2])
    graph.add_node("evaluation_node", nodes[3])
    graph.set_entry_point("router_node")
    graph.add_edge("router_node", "text_retrieval_node")
    graph.add_edge("text_retrieval_node", "answer_generation_node")
    graph.add_edge("answer_generation_node", "evaluation_node")
    graph.add_edge("evaluation_node", END)
    return graph.compile()


def invoke_paper_qa(
    query: str,
    top_k: int = 5,
    router_agent: RouterAgent | None = None,
    retrieval_agent: RetrievalAgent | None = None,
    llm_client: LLMClient | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    if router_agent is None:
        router_agent = RouterAgent(config_path=config_path)
    if retrieval_agent is None:
        retrieval_agent = RetrievalAgent(config_path=config_path)
    if llm_client is None:
        llm_client = LLMClient(config_path=config_path)

    graph = build_paper_qa_graph(
        router_agent=router_agent,
        retrieval_agent=retrieval_agent,
        llm_client=llm_client,
    )
    initial_state = create_initial_state("paper_qa", user_query=query, top_k=top_k)
    initial_state["question"] = query
    return graph.invoke(initial_state)


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
    for key in ("messages", "trace", "retrieved_chunks", "retrieved_figures", "citations", "evaluations"):
        if key in copied:
            copied[key] = list(copied[key])  # type: ignore[literal-required]
    return copied


def _append_trace(state: ResearchState, node_name: str, payload: dict[str, Any]) -> None:
    trace = list(state.get("trace") or [])
    trace.append({"node": node_name, "payload": payload})
    state["trace"] = trace

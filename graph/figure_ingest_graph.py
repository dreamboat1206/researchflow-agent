from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from graph.state import ResearchState, create_initial_state
from models.image_embedding import ImageEmbeddingModel
from models.text_embedding import TextEmbeddingModel
from observability.trace_logger import TraceLogger
from storage.qdrant_store import QdrantFigureImageStore, QdrantFigureTextStore
from storage.sqlite_store import DEFAULT_CONFIG_PATH
from tools.figure_extractor import extract_figures_from_pdf


def extract_figures_node(
    state: ResearchState,
    extract_figures_func: Callable[..., list[dict[str, Any]]] = extract_figures_from_pdf,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    figures = extract_figures_func(
        next_state["pdf_path"],
        paper_id=next_state["paper_id"],
        config_path=config_path,
        min_width=int(next_state.get("min_width") or 80),  # type: ignore[typeddict-item]
        min_height=int(next_state.get("min_height") or 80),  # type: ignore[typeddict-item]
        write_to_sqlite=True,
    )
    next_state["retrieved_figures"] = figures
    next_state["figure_count"] = len(figures)
    _append_trace(next_state, "extract_figures_node", {"figure_count": len(figures)})
    return next_state


def figure_text_vector_node(
    state: ResearchState,
    embedding_model_factory: Callable[..., Any] = TextEmbeddingModel,
    figure_store_factory: Callable[..., Any] = QdrantFigureTextStore,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    embedding_model = embedding_model_factory(config_path=config_path)
    figure_store = figure_store_factory(embedding_model=embedding_model, config_path=config_path)
    figure_store.create_collection()
    vector_count = figure_store.upsert_figures_text(next_state.get("retrieved_figures") or [])
    next_state["figure_text_vector_count"] = vector_count
    next_state["text_collection"] = figure_store.collection_name
    _append_trace(next_state, "figure_text_vector_node", {"vector_count": vector_count})
    return next_state


def figure_image_vector_node(
    state: ResearchState,
    embedding_model_factory: Callable[..., Any] = ImageEmbeddingModel,
    figure_store_factory: Callable[..., Any] = QdrantFigureImageStore,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    embedding_model = embedding_model_factory(config_path=config_path)
    figure_store = figure_store_factory(embedding_model=embedding_model, config_path=config_path)
    figure_store.create_collection()
    vector_count = figure_store.upsert_figures_image(next_state.get("retrieved_figures") or [])
    next_state["figure_image_vector_count"] = vector_count
    next_state["image_collection"] = figure_store.collection_name
    _append_trace(next_state, "figure_image_vector_node", {"vector_count": vector_count})
    return next_state


def build_figure_ingest_graph(
    extract_figures_func: Callable[..., list[dict[str, Any]]] = extract_figures_from_pdf,
    text_embedding_model_factory: Callable[..., Any] = TextEmbeddingModel,
    figure_text_store_factory: Callable[..., Any] = QdrantFigureTextStore,
    image_embedding_model_factory: Callable[..., Any] = ImageEmbeddingModel,
    figure_image_store_factory: Callable[..., Any] = QdrantFigureImageStore,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Any:
    nodes = [
        lambda state: extract_figures_node(state, extract_figures_func=extract_figures_func, config_path=config_path),
        lambda state: figure_text_vector_node(
            state,
            embedding_model_factory=text_embedding_model_factory,
            figure_store_factory=figure_text_store_factory,
            config_path=config_path,
        ),
        lambda state: figure_image_vector_node(
            state,
            embedding_model_factory=image_embedding_model_factory,
            figure_store_factory=figure_image_store_factory,
            config_path=config_path,
        ),
    ]
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialGraph(nodes)

    graph = StateGraph(ResearchState)
    for index, node in enumerate(nodes):
        graph.add_node(f"node_{index}", node)
    graph.set_entry_point("node_0")
    for index in range(len(nodes) - 1):
        graph.add_edge(f"node_{index}", f"node_{index + 1}")
    graph.add_edge(f"node_{len(nodes) - 1}", END)
    return graph.compile()


def invoke_figure_ingest(
    file_path: str,
    paper_id: int,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    min_width: int = 80,
    min_height: int = 80,
    trace_logger: TraceLogger | None = None,
    **dependencies: Any,
) -> ResearchState:
    graph = build_figure_ingest_graph(config_path=config_path, **dependencies)
    state = create_initial_state("figure_ingest", user_query=file_path)
    state["pdf_path"] = file_path
    state["paper_id"] = paper_id
    state["min_width"] = min_width  # type: ignore[typeddict-item]
    state["min_height"] = min_height  # type: ignore[typeddict-item]
    logger = trace_logger or TraceLogger(config_path=config_path)
    trace_id, started_at = logger.start_trace()
    state["run_id"] = trace_id
    try:
        result = graph.invoke(state)
        result["run_id"] = trace_id
        logger.log_graph_result(
            trace_id=trace_id,
            started_at=started_at,
            task_type="figure_ingest",
            query=file_path,
            state=result,
        )
        return result
    except Exception as error:
        logger.log_exception(
            trace_id=trace_id,
            started_at=started_at,
            task_type="figure_ingest",
            query=file_path,
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
    for key in ("messages", "trace", "retrieved_chunks", "retrieved_figures", "citations", "evaluations"):
        if key in copied:
            copied[key] = list(copied[key])  # type: ignore[literal-required]
    return copied


def _append_trace(state: ResearchState, node_name: str, payload: dict[str, Any]) -> None:
    trace = list(state.get("trace") or [])
    trace.append({"node": node_name, "payload": payload})
    state["trace"] = trace

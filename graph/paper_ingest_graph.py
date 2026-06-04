from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.organizer_agent import OrganizerAgent
from graph.state import ResearchState, create_initial_state
from models.text_embedding import TextEmbeddingModel
from observability.trace_logger import TraceLogger
from storage.qdrant_store import QdrantTextStore
from storage.sqlite_store import DEFAULT_CONFIG_PATH, init_db, insert_chunk, insert_paper
from tools.pdf_parser import parse_pdf
from tools.text_splitter import split_pages_to_chunks


def parse_pdf_node(state: ResearchState, parse_pdf_func: Callable[..., dict[str, Any]] = parse_pdf) -> ResearchState:
    next_state = _copy_state(state)
    parsed_pdf = parse_pdf_func(str(next_state["pdf_path"]))
    next_state["parsed_pdf"] = parsed_pdf
    next_state["paper_title"] = parsed_pdf.get("title")
    _append_trace(next_state, "parse_pdf_node", {"title": parsed_pdf.get("title")})
    return next_state


def sqlite_paper_node(
    state: ResearchState,
    init_db_func: Callable[..., Any] = init_db,
    insert_paper_func: Callable[..., int] = insert_paper,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    init_db_func(config_path)
    parsed_pdf = next_state["parsed_pdf"]
    paper_id = insert_paper_func(
        title=parsed_pdf.get("title") or next_state["pdf_path"],
        authors=parsed_pdf.get("authors"),
        year=parsed_pdf.get("year"),
        source_path=parsed_pdf["file_path"],
        abstract=parsed_pdf.get("abstract"),
        metadata=parsed_pdf.get("metadata"),
        config_path=config_path,
    )
    next_state["paper_id"] = paper_id
    _append_trace(next_state, "sqlite_paper_node", {"paper_id": paper_id})
    return next_state


def chunk_node(
    state: ResearchState,
    split_pages_to_chunks_func: Callable[..., list[dict[str, Any]]] = split_pages_to_chunks,
) -> ResearchState:
    next_state = _copy_state(state)
    chunks = split_pages_to_chunks_func(
        next_state["parsed_pdf"]["pages"],
        paper_id=next_state["paper_id"],
        chunk_size=int(next_state.get("chunk_size") or 1000),  # type: ignore[typeddict-item]
        overlap=int(next_state.get("overlap") or 100),  # type: ignore[typeddict-item]
    )
    next_state["chunks"] = chunks
    _append_trace(next_state, "chunk_node", {"chunk_count": len(chunks)})
    return next_state


def text_vector_node(
    state: ResearchState,
    embedding_model_factory: Callable[..., Any] = TextEmbeddingModel,
    qdrant_store_factory: Callable[..., Any] = QdrantTextStore,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    embedding_model = embedding_model_factory(config_path=config_path)
    qdrant_store = qdrant_store_factory(embedding_model=embedding_model, config_path=config_path)
    qdrant_store.create_collection()
    vector_count = qdrant_store.upsert_text_chunks(next_state["chunks"])
    next_state["text_vector_count"] = vector_count
    next_state["collection"] = qdrant_store.collection_name
    _append_trace(next_state, "text_vector_node", {"vector_count": vector_count})
    return next_state


def sqlite_chunks_node(
    state: ResearchState,
    insert_chunk_func: Callable[..., int] = insert_chunk,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    for chunk_index, chunk in enumerate(next_state.get("chunks") or []):
        insert_chunk_func(
            paper_id=next_state["paper_id"],
            chunk_index=chunk_index,
            text=chunk["chunk_text"],
            page_start=chunk["page"],
            page_end=chunk["page"],
            metadata={"chunk_id": chunk["chunk_id"]},
            config_path=config_path,
        )
    _append_trace(next_state, "sqlite_chunks_node", {"chunk_count": len(next_state.get("chunks") or [])})
    return next_state


def organize_node(
    state: ResearchState,
    organizer_factory: Callable[..., OrganizerAgent] = OrganizerAgent,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ResearchState:
    next_state = _copy_state(state)
    if not next_state.get("organize_after_ingest"):
        return next_state
    try:
        organization = organizer_factory(config_path=config_path).organize_paper(
            str(next_state["paper_id"]),
            dry_run=False,
            mode=next_state.get("organize_mode"),
        )
        next_state["organization"] = organization.to_dict()
    except Exception as error:
        next_state["organization_error"] = str(error)
    _append_trace(next_state, "organize_node", {"organized": "organization" in next_state})
    return next_state


def build_paper_ingest_graph(
    parse_pdf_func: Callable[..., dict[str, Any]] = parse_pdf,
    init_db_func: Callable[..., Any] = init_db,
    insert_paper_func: Callable[..., int] = insert_paper,
    split_pages_to_chunks_func: Callable[..., list[dict[str, Any]]] = split_pages_to_chunks,
    embedding_model_factory: Callable[..., Any] = TextEmbeddingModel,
    qdrant_store_factory: Callable[..., Any] = QdrantTextStore,
    insert_chunk_func: Callable[..., int] = insert_chunk,
    organizer_factory: Callable[..., OrganizerAgent] = OrganizerAgent,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Any:
    nodes = [
        lambda state: parse_pdf_node(state, parse_pdf_func=parse_pdf_func),
        lambda state: sqlite_paper_node(
            state,
            init_db_func=init_db_func,
            insert_paper_func=insert_paper_func,
            config_path=config_path,
        ),
        lambda state: chunk_node(state, split_pages_to_chunks_func=split_pages_to_chunks_func),
        lambda state: text_vector_node(
            state,
            embedding_model_factory=embedding_model_factory,
            qdrant_store_factory=qdrant_store_factory,
            config_path=config_path,
        ),
        lambda state: sqlite_chunks_node(state, insert_chunk_func=insert_chunk_func, config_path=config_path),
        lambda state: organize_node(state, organizer_factory=organizer_factory, config_path=config_path),
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


def invoke_paper_ingest(
    file_path: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    chunk_size: int = 1000,
    overlap: int = 100,
    organize: bool = False,
    organize_mode: str | None = None,
    trace_logger: TraceLogger | None = None,
    **dependencies: Any,
) -> ResearchState:
    graph = build_paper_ingest_graph(config_path=config_path, **dependencies)
    state = create_initial_state("paper_ingest", user_query=file_path)
    state["pdf_path"] = file_path
    state["chunk_size"] = chunk_size  # type: ignore[typeddict-item]
    state["overlap"] = overlap  # type: ignore[typeddict-item]
    state["organize_after_ingest"] = organize
    state["organize_mode"] = organize_mode
    logger = trace_logger or TraceLogger(config_path=config_path)
    trace_id, started_at = logger.start_trace()
    state["run_id"] = trace_id
    try:
        result = graph.invoke(state)
        result["run_id"] = trace_id
        logger.log_graph_result(
            trace_id=trace_id,
            started_at=started_at,
            task_type="paper_ingest",
            query=file_path,
            state=result,
        )
        return result
    except Exception as error:
        logger.log_exception(
            trace_id=trace_id,
            started_at=started_at,
            task_type="paper_ingest",
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
    for key in ("messages", "trace", "retrieved_chunks", "retrieved_figures", "citations", "evaluations", "chunks"):
        if key in copied:
            copied[key] = list(copied[key])  # type: ignore[literal-required]
    return copied


def _append_trace(state: ResearchState, node_name: str, payload: dict[str, Any]) -> None:
    trace = list(state.get("trace") or [])
    trace.append({"node": node_name, "payload": payload})
    state["trace"] = trace

import argparse
import json
import sys

from agents.organizer_agent import OrganizerAgent
from graph.figure_ingest_graph import invoke_figure_ingest
from graph.organizer_graph import invoke_organize_paper, invoke_organize_papers
from graph.paper_ingest_graph import invoke_paper_ingest
from models.text_embedding import TextEmbeddingModel
from models.image_embedding import ImageEmbeddingModel
from storage.qdrant_store import QdrantFigureImageStore, QdrantFigureTextStore, QdrantTextStore
from storage.sqlite_store import init_db, insert_chunk, insert_paper
from tools.figure_extractor import extract_figures_from_pdf
from tools.pdf_parser import parse_pdf
from tools.text_splitter import split_pages_to_chunks


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="ResearchFlow-Agent command line entry.")
    subparsers = parser.add_subparsers(dest="command")

    init_db_parser = subparsers.add_parser("init-db", help="Initialize the SQLite database.")
    init_db_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project configuration file.",
    )

    parse_pdf_parser = subparsers.add_parser("parse-pdf", help="Parse text from a PDF file.")
    parse_pdf_parser.add_argument("file_path", help="Path to the PDF file.")

    ingest_pdf_parser = subparsers.add_parser(
        "ingest-pdf",
        help="Parse, chunk, embed, and store a PDF in SQLite and Qdrant.",
    )
    ingest_pdf_parser.add_argument("file_path", help="Path to the PDF file.")
    ingest_pdf_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project configuration file.",
    )
    ingest_pdf_parser.add_argument("--chunk-size", type=int, default=1000)
    ingest_pdf_parser.add_argument("--overlap", type=int, default=100)
    ingest_pdf_parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize the PDF into the structured library after ingest.",
    )
    ingest_pdf_parser.add_argument(
        "--organize-mode",
        choices=("copy", "move"),
        default=None,
        help="File operation mode for --organize. Defaults to organizer.default_mode.",
    )

    organize_paper_parser = subparsers.add_parser(
        "organize-paper",
        help="Classify and organize one ingested paper.",
    )
    organize_paper_parser.add_argument("paper_id", help="SQLite paper id.")
    organize_paper_parser.add_argument("--config", default="config.yaml")
    organize_paper_parser.add_argument("--dry-run", action="store_true", help="Plan without copying or moving.")
    organize_paper_parser.add_argument("--apply", action="store_true", help="Apply file and SQLite changes.")
    organize_paper_parser.add_argument("--mode", choices=("copy", "move"), default=None)

    organize_papers_parser = subparsers.add_parser(
        "organize-papers",
        help="Classify and organize ingested papers.",
    )
    organize_papers_parser.add_argument("--config", default="config.yaml")
    organize_papers_parser.add_argument("--dry-run", action="store_true", help="Plan without copying or moving.")
    organize_papers_parser.add_argument("--apply", action="store_true", help="Apply file and SQLite changes.")
    organize_papers_parser.add_argument("--mode", choices=("copy", "move"), default=None)
    organize_papers_parser.add_argument("--limit", type=int, default=None)

    extract_figures_parser = subparsers.add_parser(
        "extract-figures",
        help="Extract embedded PDF images and store figure metadata.",
    )
    extract_figures_parser.add_argument("file_path", help="Path to the PDF file.")
    extract_figures_parser.add_argument("paper_id", type=int, help="SQLite paper id.")
    extract_figures_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project configuration file.",
    )
    extract_figures_parser.add_argument("--min-width", type=int, default=80)
    extract_figures_parser.add_argument("--min-height", type=int, default=80)

    parser.add_argument(
        "--version",
        action="version",
        version="ResearchFlow-Agent 0.1.0",
    )
    args = parser.parse_args()

    if args.command == "init-db":
        db_path = init_db(args.config)
        print(f"Initialized SQLite database at {db_path}")
        return

    if args.command == "parse-pdf":
        result = parse_pdf(args.file_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "ingest-pdf":
        result = ingest_pdf(
            args.file_path,
            config_path=args.config,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            organize=args.organize,
            organize_mode=args.organize_mode,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "organize-paper":
        graph_state = invoke_organize_paper(
            args.paper_id,
            dry_run=not args.apply,
            mode=args.mode,
            config_path=args.config,
        )
        print(json.dumps(graph_state.get("organization_result", {}), ensure_ascii=False, indent=2))
        return

    if args.command == "organize-papers":
        graph_state = invoke_organize_papers(
            dry_run=not args.apply,
            mode=args.mode,
            limit=args.limit,
            config_path=args.config,
        )
        print(json.dumps(graph_state.get("organization_results", []), ensure_ascii=False, indent=2))
        return

    if args.command == "extract-figures":
        graph_state = invoke_figure_ingest(
            args.file_path,
            paper_id=args.paper_id,
            config_path=args.config,
            min_width=args.min_width,
            min_height=args.min_height,
            extract_figures_func=extract_figures_from_pdf,
            text_embedding_model_factory=TextEmbeddingModel,
            figure_text_store_factory=QdrantFigureTextStore,
            image_embedding_model_factory=ImageEmbeddingModel,
            figure_image_store_factory=QdrantFigureImageStore,
        )
        print(
            json.dumps(
                {
                    "figures": graph_state.get("retrieved_figures", []),
                    "count": graph_state.get("figure_count", 0),
                    "text_vectors": graph_state.get("figure_text_vector_count", 0),
                    "text_collection": graph_state.get("text_collection"),
                    "image_vectors": graph_state.get("figure_image_vector_count", 0),
                    "image_collection": graph_state.get("image_collection"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    parser.print_help()


def ingest_pdf(
    file_path: str,
    config_path: str = "config.yaml",
    chunk_size: int = 1000,
    overlap: int = 100,
    organize: bool = False,
    organize_mode: str | None = None,
) -> dict[str, object]:
    graph_state = invoke_paper_ingest(
        file_path,
        config_path=config_path,
        chunk_size=chunk_size,
        overlap=overlap,
        organize=organize,
        organize_mode=organize_mode,
        parse_pdf_func=parse_pdf,
        init_db_func=init_db,
        insert_paper_func=insert_paper,
        split_pages_to_chunks_func=split_pages_to_chunks,
        embedding_model_factory=TextEmbeddingModel,
        qdrant_store_factory=QdrantTextStore,
        insert_chunk_func=insert_chunk,
        organizer_factory=OrganizerAgent,
    )
    result: dict[str, object] = {
        "paper_id": graph_state.get("paper_id"),
        "chunks": len(graph_state.get("chunks") or []),
        "vectors": graph_state.get("text_vector_count", 0),
        "collection": graph_state.get("collection"),
    }
    if graph_state.get("organization"):
        result["organization"] = graph_state["organization"]
    if graph_state.get("organization_error"):
        result["organization_error"] = graph_state["organization_error"]
    return result

if __name__ == "__main__":
    main()

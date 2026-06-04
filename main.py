import argparse
import json
import sys

from agents.organizer_agent import OrganizerAgent, PaperOrganizationResult
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
        agent = OrganizerAgent(config_path=args.config)
        result = agent.organize_paper(
            args.paper_id,
            dry_run=not args.apply,
            mode=args.mode,
        )
        print(json.dumps(_organization_result_to_dict(result), ensure_ascii=False, indent=2))
        return

    if args.command == "organize-papers":
        agent = OrganizerAgent(config_path=args.config)
        results = agent.organize_all_papers(
            dry_run=not args.apply,
            mode=args.mode,
            limit=args.limit,
        )
        print(json.dumps([_organization_result_to_dict(result) for result in results], ensure_ascii=False, indent=2))
        return

    if args.command == "extract-figures":
        figures = extract_figures_from_pdf(
            args.file_path,
            paper_id=args.paper_id,
            config_path=args.config,
            min_width=args.min_width,
            min_height=args.min_height,
            write_to_sqlite=True,
        )
        embedding_model = TextEmbeddingModel(config_path=args.config)
        figure_store = QdrantFigureTextStore(embedding_model=embedding_model, config_path=args.config)
        figure_store.create_collection()
        vector_count = figure_store.upsert_figures_text(figures)
        image_embedding_model = ImageEmbeddingModel(config_path=args.config)
        figure_image_store = QdrantFigureImageStore(
            embedding_model=image_embedding_model,
            config_path=args.config,
        )
        figure_image_store.create_collection()
        image_vector_count = figure_image_store.upsert_figures_image(figures)
        print(
            json.dumps(
                {
                    "figures": figures,
                    "count": len(figures),
                    "text_vectors": vector_count,
                    "text_collection": figure_store.collection_name,
                    "image_vectors": image_vector_count,
                    "image_collection": figure_image_store.collection_name,
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
    init_db(config_path)
    parsed_pdf = parse_pdf(file_path)
    paper_id = insert_paper(
        title=parsed_pdf.get("title") or file_path,
        authors=parsed_pdf.get("authors"),
        year=parsed_pdf.get("year"),
        source_path=parsed_pdf["file_path"],
        abstract=parsed_pdf.get("abstract"),
        metadata=parsed_pdf.get("metadata"),
        config_path=config_path,
    )
    chunks = split_pages_to_chunks(
        parsed_pdf["pages"],
        paper_id=paper_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    embedding_model = TextEmbeddingModel(config_path=config_path)
    qdrant_store = QdrantTextStore(embedding_model=embedding_model, config_path=config_path)
    qdrant_store.create_collection()
    vector_count = qdrant_store.upsert_text_chunks(chunks)

    for chunk_index, chunk in enumerate(chunks):
        insert_chunk(
            paper_id=paper_id,
            chunk_index=chunk_index,
            text=chunk["chunk_text"],
            page_start=chunk["page"],
            page_end=chunk["page"],
            metadata={"chunk_id": chunk["chunk_id"]},
            config_path=config_path,
        )

    result: dict[str, object] = {
        "paper_id": paper_id,
        "chunks": len(chunks),
        "vectors": vector_count,
        "collection": qdrant_store.collection_name,
    }
    if organize:
        try:
            organization = OrganizerAgent(config_path=config_path).organize_paper(
                str(paper_id),
                dry_run=False,
                mode=organize_mode,
            )
            result["organization"] = _organization_result_to_dict(organization)
        except Exception as error:
            result["organization_error"] = str(error)
    return result


def _organization_result_to_dict(result: PaperOrganizationResult) -> dict[str, object]:
    return result.to_dict()


if __name__ == "__main__":
    main()

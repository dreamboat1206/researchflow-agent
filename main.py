import argparse
import json
import sys

from models.text_embedding import TextEmbeddingModel
from storage.qdrant_store import QdrantTextStore
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
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
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
        print(json.dumps({"figures": figures, "count": len(figures)}, ensure_ascii=False, indent=2))
        return

    parser.print_help()


def ingest_pdf(
    file_path: str,
    config_path: str = "config.yaml",
    chunk_size: int = 1000,
    overlap: int = 100,
) -> dict[str, int | str]:
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

    return {
        "paper_id": paper_id,
        "chunks": len(chunks),
        "vectors": vector_count,
        "collection": qdrant_store.collection_name,
    }


if __name__ == "__main__":
    main()

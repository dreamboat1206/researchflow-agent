import argparse
import json

from storage.sqlite_store import init_db
from tools.pdf_parser import parse_pdf


def main() -> None:
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

    parser.print_help()


if __name__ == "__main__":
    main()

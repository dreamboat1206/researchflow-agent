import argparse

from storage.sqlite_store import init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchFlow-Agent command line entry.")
    subparsers = parser.add_subparsers(dest="command")

    init_db_parser = subparsers.add_parser("init-db", help="Initialize the SQLite database.")
    init_db_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project configuration file.",
    )

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

    parser.print_help()


if __name__ == "__main__":
    main()

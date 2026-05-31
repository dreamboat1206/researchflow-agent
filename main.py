import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchFlow-Agent command line entry.")
    parser.add_argument(
        "--version",
        action="version",
        version="ResearchFlow-Agent 0.1.0",
    )
    parser.parse_args()
    print("ResearchFlow-Agent CLI placeholder. Use --help for options.")


if __name__ == "__main__":
    main()


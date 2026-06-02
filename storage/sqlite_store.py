from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    _apply_env_overrides(config)
    return config


def get_db_path(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)
    database_path = config.get("database", {}).get("path")
    if not database_path:
        raise ValueError("Missing database.path in config.yaml")

    path = Path(database_path)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path


def connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = MEMORY")
    return connection


def init_db(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    db_path = get_db_path(config_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_empty_failed_db(db_path)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(db_path) as connection:
        connection.executescript(schema)

    return db_path


def insert_paper(
    title: str,
    authors: list[str] | str | None = None,
    year: int | None = None,
    source_path: str | None = None,
    abstract: str | None = None,
    metadata: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    db_path = get_db_path(config_path)
    authors_value = json.dumps(authors, ensure_ascii=False) if isinstance(authors, list) else authors

    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO papers (title, authors, year, source_path, abstract, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                authors_value,
                year,
                source_path,
                abstract,
                _to_json(metadata),
            ),
        )
        return int(cursor.lastrowid)


def insert_chunk(
    paper_id: int,
    chunk_index: int,
    text: str,
    page_start: int | None = None,
    page_end: int | None = None,
    metadata: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    db_path = get_db_path(config_path)

    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO chunks (paper_id, chunk_index, text, page_start, page_end, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (paper_id, chunk_index, text, page_start, page_end, _to_json(metadata)),
        )
        return int(cursor.lastrowid)


def insert_figure(
    paper_id: int,
    figure_index: int,
    page_number: int | None = None,
    caption: str | None = None,
    image_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    db_path = get_db_path(config_path)

    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO figures (paper_id, figure_index, page_number, caption, image_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (paper_id, figure_index, page_number, caption, image_path, _to_json(metadata)),
        )
        return int(cursor.lastrowid)


def list_papers(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)

    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, title, authors, year, source_path, abstract, metadata, created_at
            FROM papers
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()

    return [_paper_from_row(row) for row in rows]


def get_paper(paper_id: int, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any] | None:
    db_path = get_db_path(config_path)

    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, title, authors, year, source_path, abstract, metadata, created_at
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        ).fetchone()

    if row is None:
        return None
    return _paper_from_row(row)


def _to_json(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _apply_env_overrides(config: dict[str, Any]) -> None:
    _set_if_env(config, ("database", "path"), "DATABASE_PATH")
    _set_if_env(config, ("qdrant", "url"), "QDRANT_URL")
    _set_if_env(config, ("qdrant", "collection_name"), "QDRANT_COLLECTION")
    _set_if_env(config, ("models", "embedding_model"), "EMBEDDING_MODEL")
    _set_bool_if_env(config, ("models", "embedding_local_files_only"), "EMBEDDING_LOCAL_FILES_ONLY")
    _set_if_env(config, ("models", "llm_model"), "LLM_MODEL")
    _set_if_env(config, ("llm", "provider"), "LLM_PROVIDER")
    _set_if_env(config, ("llm", "base_url"), "OPENAI_BASE_URL")
    _set_if_env(config, ("llm", "api_key"), "OPENAI_API_KEY")
    _set_if_env(config, ("api", "host"), "API_HOST")
    _set_int_if_env(config, ("api", "port"), "API_PORT")


def _set_if_env(config: dict[str, Any], path: tuple[str, str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None:
        return
    section, key = path
    config.setdefault(section, {})[key] = value


def _set_bool_if_env(config: dict[str, Any], path: tuple[str, str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None:
        return
    section, key = path
    config.setdefault(section, {})[key] = value.lower() in {"1", "true", "yes", "on"}


def _set_int_if_env(config: dict[str, Any], path: tuple[str, str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None:
        return
    section, key = path
    config.setdefault(section, {})[key] = int(value)


def _remove_empty_failed_db(db_path: Path) -> None:
    journal_path = db_path.with_name(f"{db_path.name}-journal")
    if db_path.exists() and db_path.stat().st_size == 0 and journal_path.exists():
        journal_path.unlink()
        db_path.unlink()


def _from_json(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _paper_from_row(row: sqlite3.Row) -> dict[str, Any]:
    authors = row["authors"]
    if authors and authors.startswith("["):
        authors = json.loads(authors)

    return {
        "id": row["id"],
        "title": row["title"],
        "authors": authors,
        "year": row["year"],
        "source_path": row["source_path"],
        "abstract": row["abstract"],
        "metadata": _from_json(row["metadata"]),
        "created_at": row["created_at"],
    }

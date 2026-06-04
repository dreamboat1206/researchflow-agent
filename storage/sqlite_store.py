from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from models.figure_record import FigureRecord, FigureType, build_figure_id


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
        _ensure_paper_columns(connection)
        _ensure_organizer_tables(connection)

    return db_path


def insert_paper(
    title: str,
    authors: list[str] | str | None = None,
    year: int | None = None,
    source_path: str | None = None,
    original_path: str | None = None,
    abstract: str | None = None,
    metadata: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    db_path = get_db_path(config_path)
    authors_value = json.dumps(authors, ensure_ascii=False) if isinstance(authors, list) else authors

    with connect(db_path) as connection:
        _ensure_paper_columns(connection)
        cursor = connection.execute(
            """
            INSERT INTO papers (title, authors, year, source_path, original_path, abstract, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                authors_value,
                year,
                source_path,
                original_path or source_path,
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
    nearby_text: str | None = None,
    image_path: str | None = None,
    figure_type: str | FigureType = FigureType.OTHER,
    figure_id: str | None = None,
    page: int | None = None,
    metadata: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    db_path = get_db_path(config_path)
    page_value = page if page is not None else page_number
    if page_value is None:
        raise ValueError("page or page_number is required for figure metadata")
    if not image_path:
        raise ValueError("image_path is required for figure metadata")

    figure_type_value = FigureType(figure_type).value
    figure_id_value = figure_id or build_figure_id(paper_id, page_value, figure_index)

    with connect(db_path) as connection:
        _ensure_figure_columns(connection)
        existing_row = connection.execute(
            "SELECT id FROM figures WHERE figure_id = ?",
            (figure_id_value,),
        ).fetchone()
        if existing_row is not None:
            connection.execute(
                """
                UPDATE figures
                SET paper_id = ?, figure_index = ?, page = ?, page_number = ?, figure_type = ?,
                    caption = ?, nearby_text = ?, image_path = ?, metadata = ?
                WHERE figure_id = ?
                """,
                (
                    paper_id,
                    figure_index,
                    page_value,
                    page_value,
                    figure_type_value,
                    caption,
                    nearby_text,
                    image_path,
                    _to_json(metadata),
                    figure_id_value,
                ),
            )
            return int(existing_row["id"])

        cursor = connection.execute(
            """
            INSERT INTO figures (
                figure_id, paper_id, figure_index, page, page_number, figure_type,
                caption, nearby_text, image_path, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                figure_id_value,
                paper_id,
                figure_index,
                page_value,
                page_value,
                figure_type_value,
                caption,
                nearby_text,
                image_path,
                _to_json(metadata),
            ),
        )
        return int(cursor.lastrowid)


def insert_figure_record(
    record: FigureRecord,
    metadata: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    return insert_figure(
        paper_id=record.paper_id,
        figure_index=record.figure_index,
        page=record.page,
        caption=record.caption,
        nearby_text=None,
        image_path=record.image_path,
        figure_type=record.figure_type,
        figure_id=record.figure_id,
        metadata=metadata,
        config_path=config_path,
    )


def update_figure_caption(
    figure_id: str,
    caption: str | None,
    nearby_text: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_figure_columns(connection)
        connection.execute(
            """
            UPDATE figures
            SET caption = ?, nearby_text = ?
            WHERE figure_id = ?
            """,
            (caption, nearby_text, figure_id),
        )


def list_figures(
    paper_id: int | None = None,
    title_query: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)
    where_clauses: list[str] = []
    parameters: list[Any] = []
    if paper_id is not None:
        where_clauses.append("figures.paper_id = ?")
        parameters.append(paper_id)
    if title_query:
        where_clauses.append("LOWER(papers.title) LIKE ?")
        parameters.append(f"%{title_query.lower()}%")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with connect(db_path) as connection:
        _ensure_figure_columns(connection)
        rows = connection.execute(
            f"""
            SELECT figures.id, figures.figure_id, figures.paper_id, papers.title AS paper_title,
                   figures.figure_index, figures.page, figures.page_number, figures.figure_type,
                   figures.caption, figures.nearby_text, figures.image_path, figures.metadata,
                   figures.created_at
            FROM figures
            LEFT JOIN papers ON papers.id = figures.paper_id
            {where_sql}
            ORDER BY figures.paper_id ASC, figures.page ASC, figures.figure_index ASC, figures.id ASC
            """,
            parameters,
        ).fetchall()
    return [_figure_from_row(row) for row in rows]


def get_figure_by_id(
    figure_id: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any] | None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_figure_columns(connection)
        row = connection.execute(
            """
            SELECT figures.id, figures.figure_id, figures.paper_id, papers.title AS paper_title,
                   figures.figure_index, figures.page, figures.page_number, figures.figure_type,
                   figures.caption, figures.nearby_text, figures.image_path, figures.metadata,
                   figures.created_at
            FROM figures
            LEFT JOIN papers ON papers.id = figures.paper_id
            WHERE figures.figure_id = ?
            """,
            (figure_id,),
        ).fetchone()
    if row is None:
        return None
    return _figure_from_row(row)


def list_papers(
    limit: int | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)
    limit_sql = "LIMIT ?" if limit is not None else ""
    parameters: tuple[Any, ...] = (limit,) if limit is not None else ()

    with connect(db_path) as connection:
        _ensure_paper_columns(connection)
        rows = connection.execute(
            f"""
            SELECT id, title, authors, year, source_path, original_path, organized_path,
                   paper_type, primary_topic, organized_at, organization_status,
                   filename_title_source, abstract, metadata, created_at
            FROM papers
            ORDER BY created_at DESC, id DESC
            {limit_sql}
            """,
            parameters,
        ).fetchall()

    return [_paper_from_row(row) for row in rows]


def get_paper(paper_id: int, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any] | None:
    db_path = get_db_path(config_path)

    with connect(db_path) as connection:
        _ensure_paper_columns(connection)
        row = connection.execute(
            """
            SELECT id, title, authors, year, source_path, original_path, organized_path,
                   paper_type, primary_topic, organized_at, organization_status,
                   filename_title_source, abstract, metadata, created_at
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        ).fetchone()

    if row is None:
        return None
    return _paper_from_row(row)


def get_paper_by_id(paper_id: int | str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any] | None:
    return get_paper(int(paper_id), config_path=config_path)


def get_chunks_for_paper(
    paper_id: int | str,
    limit: int = 10,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, paper_id, chunk_index, text, page_start, page_end, metadata, created_at
            FROM chunks
            WHERE paper_id = ?
            ORDER BY chunk_index ASC, id ASC
            LIMIT ?
            """,
            (int(paper_id), limit),
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def update_paper_organization(
    paper_id: int | str,
    paper_type: str | None = None,
    primary_topic: str | None = None,
    year: int | None = None,
    original_path: str | None = None,
    organized_path: str | None = None,
    filename_title_source: str | None = None,
    organization_status: str = "organized",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_paper_columns(connection)
        connection.execute(
            """
            UPDATE papers
            SET paper_type = COALESCE(?, paper_type),
                primary_topic = COALESCE(?, primary_topic),
                year = COALESCE(?, year),
                original_path = COALESCE(?, original_path),
                organized_path = COALESCE(?, organized_path),
                filename_title_source = COALESCE(?, filename_title_source),
                organization_status = ?,
                organized_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                paper_type,
                primary_topic,
                year,
                original_path,
                organized_path,
                filename_title_source,
                organization_status,
                int(paper_id),
            ),
        )


def insert_paper_tag(
    paper_id: int | str,
    tag: str,
    tag_type: str,
    confidence: float = 1.0,
    source: str = "organizer",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO paper_tags (paper_id, tag, tag_type, confidence, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(paper_id), tag, tag_type, confidence, source),
        )


def insert_paper_tags(
    paper_id: int | str,
    tags: list[dict[str, Any]],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    for tag in tags:
        insert_paper_tag(
            paper_id,
            str(tag["tag"]),
            str(tag["tag_type"]),
            float(tag.get("confidence", 1.0)),
            str(tag.get("source", "organizer")),
            config_path=config_path,
        )


def list_paper_tags(
    paper_id: int | str | None = None,
    tag_type: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)
    where_clauses: list[str] = []
    parameters: list[Any] = []
    if paper_id is not None:
        where_clauses.append("paper_id = ?")
        parameters.append(str(paper_id))
    if tag_type is not None:
        where_clauses.append("tag_type = ?")
        parameters.append(tag_type)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        rows = connection.execute(
            f"""
            SELECT id, paper_id, tag, tag_type, confidence, source, created_at
            FROM paper_tags
            {where_sql}
            ORDER BY paper_id ASC, tag_type ASC, tag ASC
            """,
            parameters,
        ).fetchall()
    return [dict(row) for row in rows]


def create_collection(
    collection_id: str,
    name: str,
    description: str = "",
    rule: str = "",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO paper_collections (collection_id, name, description, rule)
            VALUES (?, ?, ?, ?)
            """,
            (collection_id, name, description, rule),
        )


def add_paper_to_collection(
    collection_id: str,
    paper_id: int | str,
    reason: str = "",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO paper_collection_items (collection_id, paper_id, reason)
            VALUES (?, ?, ?)
            """,
            (collection_id, str(paper_id), reason),
        )


def list_collections(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        rows = connection.execute(
            """
            SELECT collection_id, name, description, rule, created_at
            FROM paper_collections
            ORDER BY collection_id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_collection_items(
    collection_id: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        rows = connection.execute(
            """
            SELECT collection_id, paper_id, reason, created_at
            FROM paper_collection_items
            WHERE collection_id = ?
            ORDER BY paper_id ASC
            """,
            (collection_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_organizer_results(
    paper_id: int | str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    db_path = get_db_path(config_path)
    with connect(db_path) as connection:
        _ensure_organizer_tables(connection)
        if paper_id is None:
            connection.execute("DELETE FROM paper_tags WHERE source = 'organizer'")
            connection.execute("DELETE FROM paper_collection_items")
            connection.execute(
                """
                UPDATE papers
                SET organized_path = NULL,
                    paper_type = NULL,
                    primary_topic = NULL,
                    organized_at = NULL,
                    organization_status = 'pending',
                    filename_title_source = NULL
                """
            )
            return
        connection.execute(
            "DELETE FROM paper_tags WHERE paper_id = ? AND source = 'organizer'",
            (str(paper_id),),
        )
        connection.execute(
            "DELETE FROM paper_collection_items WHERE paper_id = ?",
            (str(paper_id),),
        )
        connection.execute(
            """
            UPDATE papers
            SET organized_path = NULL,
                paper_type = NULL,
                primary_topic = NULL,
                organized_at = NULL,
                organization_status = 'pending',
                filename_title_source = NULL
            WHERE id = ?
            """,
            (int(paper_id),),
        )


def _to_json(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _apply_env_overrides(config: dict[str, Any]) -> None:
    _set_if_env(config, ("database", "path"), "DATABASE_PATH")
    _set_if_env(config, ("qdrant", "url"), "QDRANT_URL")
    _set_if_env(config, ("qdrant", "collection_name"), "QDRANT_COLLECTION")
    _set_if_env(config, ("qdrant", "figures_collection_name"), "QDRANT_FIGURES_COLLECTION")
    _set_if_env(config, ("qdrant", "figures_image_collection_name"), "QDRANT_FIGURES_IMAGE_COLLECTION")
    _set_if_env(config, ("models", "embedding_model"), "EMBEDDING_MODEL")
    _set_bool_if_env(config, ("models", "embedding_local_files_only"), "EMBEDDING_LOCAL_FILES_ONLY")
    _set_if_env(config, ("models", "image_embedding_model"), "IMAGE_EMBEDDING_MODEL")
    _set_if_env(config, ("models", "image_embedding_pretrained"), "IMAGE_EMBEDDING_PRETRAINED")
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


def _ensure_figure_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(figures)").fetchall()
    }
    if "figure_id" not in columns:
        connection.execute("ALTER TABLE figures ADD COLUMN figure_id TEXT")
    if "page" not in columns:
        connection.execute("ALTER TABLE figures ADD COLUMN page INTEGER")
    if "figure_type" not in columns:
        connection.execute("ALTER TABLE figures ADD COLUMN figure_type TEXT NOT NULL DEFAULT 'other'")
    if "nearby_text" not in columns:
        connection.execute("ALTER TABLE figures ADD COLUMN nearby_text TEXT")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_figures_figure_id_unique ON figures (figure_id)")


def _ensure_paper_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(papers)").fetchall()
    }
    additions = {
        "original_path": "TEXT",
        "organized_path": "TEXT",
        "paper_type": "TEXT",
        "primary_topic": "TEXT",
        "organized_at": "TEXT",
        "organization_status": "TEXT DEFAULT 'pending'",
        "filename_title_source": "TEXT",
    }
    for column_name, column_type in additions.items():
        if column_name not in columns:
            connection.execute(f"ALTER TABLE papers ADD COLUMN {column_name} {column_type}")
    connection.execute(
        """
        UPDATE papers
        SET original_path = source_path
        WHERE original_path IS NULL AND source_path IS NOT NULL
        """
    )
    connection.execute(
        """
        UPDATE papers
        SET organization_status = 'pending'
        WHERE organization_status IS NULL
        """
    )


def _ensure_organizer_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            tag_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT 'organizer',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(paper_id, tag, tag_type)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_collections (
            collection_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            rule TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_collection_items (
            collection_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(collection_id, paper_id),
            FOREIGN KEY (collection_id) REFERENCES paper_collections (collection_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_paper_tags_paper_id ON paper_tags (paper_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_paper_tags_tag_type ON paper_tags (tag_type)")


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
        "original_path": _row_value(row, "original_path"),
        "organized_path": _row_value(row, "organized_path"),
        "paper_type": _row_value(row, "paper_type"),
        "primary_topic": _row_value(row, "primary_topic"),
        "organized_at": _row_value(row, "organized_at"),
        "organization_status": _row_value(row, "organization_status"),
        "filename_title_source": _row_value(row, "filename_title_source"),
        "abstract": row["abstract"],
        "metadata": _from_json(row["metadata"]),
        "created_at": row["created_at"],
    }


def _chunk_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "paper_id": row["paper_id"],
        "chunk_index": row["chunk_index"],
        "text": row["text"],
        "chunk_text": row["text"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "metadata": _from_json(row["metadata"]),
        "created_at": row["created_at"],
    }


def _figure_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "figure_id": row["figure_id"],
        "paper_id": row["paper_id"],
        "paper_title": row["paper_title"] if "paper_title" in row.keys() else None,
        "figure_index": row["figure_index"],
        "page": row["page"],
        "page_number": row["page_number"],
        "figure_type": row["figure_type"],
        "caption": row["caption"],
        "nearby_text": row["nearby_text"],
        "image_path": row["image_path"],
        "metadata": _from_json(row["metadata"]),
        "created_at": row["created_at"],
    }


def _row_value(row: sqlite3.Row, key: str) -> Any:
    if key not in row.keys():
        return None
    return row[key]

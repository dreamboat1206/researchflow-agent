CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    authors TEXT,
    year INTEGER,
    source_path TEXT,
    original_path TEXT,
    organized_path TEXT,
    paper_type TEXT,
    primary_topic TEXT,
    organized_at TEXT,
    organization_status TEXT DEFAULT 'pending',
    filename_title_source TEXT,
    abstract TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS figures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figure_id TEXT UNIQUE,
    paper_id INTEGER NOT NULL,
    figure_index INTEGER NOT NULL,
    page INTEGER,
    page_number INTEGER,
    figure_type TEXT NOT NULL DEFAULT 'other',
    caption TEXT,
    nearby_text TEXT,
    image_path TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    tag_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'organizer',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, tag, tag_type)
);

CREATE TABLE IF NOT EXISTS paper_collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    rule TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_collection_items (
    collection_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(collection_id, paper_id),
    FOREIGN KEY (collection_id) REFERENCES paper_collections (collection_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_paper_id ON figures (paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_figure_id ON figures (figure_id);
CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces (run_id);
CREATE INDEX IF NOT EXISTS idx_paper_tags_paper_id ON paper_tags (paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_tags_tag_type ON paper_tags (tag_type);


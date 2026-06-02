CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    authors TEXT,
    year INTEGER,
    source_path TEXT,
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

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_paper_id ON figures (paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_figure_id ON figures (figure_id);
CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces (run_id);


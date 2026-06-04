# ResearchFlow-Agent

ResearchFlow-Agent is an agent-oriented research workflow project. The initial scaffold provides a FastAPI backend, a Streamlit UI entry point, SQLite metadata storage, configuration files, storage folders, and a basic test suite.

## Tech Stack

- FastAPI for the API service
- Streamlit for the lightweight user interface
- Pydantic for data validation
- SQLite for local metadata storage
- PyMuPDF for PDF parsing
- Qdrant client for vector database access
- sentence-transformers for embeddings
- loguru for logging
- pytest for tests

## Directory Structure

```text
.
|-- agents/              # Agent implementations
|-- api/                 # FastAPI server and routes
|-- app/                 # Streamlit application
|-- data/                # Local data and uploaded files
|-- evals/               # Evaluation scripts and datasets
|-- graph/               # Workflow graph definitions
|-- models/              # Domain and ML model helpers
|-- observability/       # Logging, tracing, and monitoring helpers
|-- storage/             # Persistence and vector store integrations
|-- tests/               # Automated tests
|-- tools/               # Agent tools
|-- config.yaml          # Project configuration
|-- main.py              # CLI entry point
`-- requirements.txt     # Python dependencies
```

## Database

The local metadata store uses SQLite. The database path is read from `config.yaml`:

```yaml
database:
  path: data/researchflow.db
```

Initialize the database:

```bash
python main.py init-db
```

The schema is defined in `storage/schema.sql` and currently includes:

- `papers` for paper-level metadata
- `chunks` for text chunks linked to papers
- `figures` for extracted figure/table metadata linked to papers
- `traces` for future agent and retrieval traces

## Paper Organizer / 论文自动整理

The paper organizer classifies ingested papers and plans or applies a structured
library path:

```text
data/library/{primary_topic}/{year}/{paper_title}.pdf
```

The final file name comes from the paper title already extracted by the PDF
parser and stored in SQLite `papers.title`. It does not re-parse the PDF title
during organization. If the stored title is not usable, it falls back to the
original PDF file stem.

Dry-run mode only prints the classification and target path:

```bash
python main.py organize-papers --dry-run
python main.py organize-paper PAPER_ID --dry-run
```

Apply mode writes organizer metadata to SQLite and copies or moves the PDF:

```bash
python main.py organize-paper PAPER_ID --apply --mode copy
python main.py organize-paper PAPER_ID --apply --mode move
```

`copy` keeps the original PDF. `move` moves the original PDF to the organized
library path. The default mode is `copy`.

You can also organize immediately after ingest:

```bash
python main.py ingest-pdf data/inbox/example.pdf --organize
```

SQLite stores organizer output on `papers`:

- `original_path`
- `organized_path`
- `paper_type`
- `primary_topic`
- `year`
- `filename_title_source`
- `organization_status`

Organizer-generated tags are stored in `paper_tags`, and automatic groupings are
stored in `paper_collections` and `paper_collection_items`.

## Figure And Table Metadata

Figure and table metadata uses `models.figure_record.FigureRecord`. The stable
`figure_id` rule is:

```text
{paper_id}_fig_{page}_{index}
```

Supported `figure_type` values are:

- `architecture`
- `pipeline`
- `chart`
- `table`
- `ablation`
- `result`
- `dataset`
- `other`

The SQLite `figures` table stores:

- `figure_id`
- `paper_id`
- `figure_index`
- `page`
- `page_number` for backward compatibility
- `figure_type`
- `caption`
- `image_path`
- `metadata`

Extracted images should be saved under:

```text
data/figures/
```

## Chunking

PDF page text can be split with `tools.text_splitter.split_pages_to_chunks`. The splitter works page by page with a fixed character window and configurable overlap. Each chunk keeps:

- `paper_id` from the caller
- stable readable `chunk_id` in the form `paper_id-p{page}-c{index}`
- source `page`
- normalized `chunk_text`

Empty pages and empty chunk text are skipped. `overlap` must be smaller than `chunk_size`.

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Initialize the SQLite database:

```bash
python main.py init-db
```

Start the API:

```bash
uvicorn api.server:app --reload
```

Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest
```

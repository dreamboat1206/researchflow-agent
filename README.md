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
- `figures` for extracted figure metadata linked to papers
- `traces` for future agent and retrieval traces

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

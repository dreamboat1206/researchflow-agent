from __future__ import annotations

from graph.paper_ingest_graph import invoke_paper_ingest


def fake_parse_pdf(file_path: str) -> dict[str, object]:
    return {
        "file_path": file_path,
        "title": "Parsed Paper",
        "authors": ["Alice"],
        "year": 2026,
        "abstract": "Abstract.",
        "metadata": {},
        "pages": [{"page": 1, "text": "Transformer method.", "file_path": file_path}],
    }


def fake_split_pages_to_chunks(pages, paper_id, chunk_size, overlap):
    return [{"paper_id": paper_id, "chunk_id": "7-p1-c1", "page": 1, "chunk_text": "Transformer method."}]


class FakeQdrantStore:
    collection_name = "papers_text"

    def __init__(self, embedding_model, config_path):
        self.embedding_model = embedding_model
        self.config_path = config_path

    def create_collection(self) -> None:
        return None

    def upsert_text_chunks(self, chunks) -> int:
        return len(chunks)


class FakeOrganizer:
    def __init__(self, config_path):
        self.config_path = config_path

    def organize_paper(self, paper_id: str, dry_run: bool, mode: str | None):
        return _FakeOrganization(paper_id)


class _FakeOrganization:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id

    def to_dict(self) -> dict[str, object]:
        return {"paper_id": self.paper_id, "organized_path": "data/library/transformer/2026/Parsed_Paper.pdf"}


def test_invoke_paper_ingest_graph_runs_all_nodes() -> None:
    paper_calls = []
    chunk_calls = []

    result = invoke_paper_ingest(
        "paper.pdf",
        config_path="config.yaml",
        organize=True,
        organize_mode="copy",
        parse_pdf_func=fake_parse_pdf,
        init_db_func=lambda config_path: None,
        insert_paper_func=lambda **kwargs: paper_calls.append(kwargs) or 7,
        split_pages_to_chunks_func=fake_split_pages_to_chunks,
        embedding_model_factory=lambda config_path: object(),
        qdrant_store_factory=FakeQdrantStore,
        insert_chunk_func=lambda **kwargs: chunk_calls.append(kwargs) or 1,
        organizer_factory=FakeOrganizer,
    )

    assert result["paper_id"] == 7
    assert result["paper_title"] == "Parsed Paper"
    assert result["text_vector_count"] == 1
    assert result["collection"] == "papers_text"
    assert result["organization"]["paper_id"] == "7"
    assert paper_calls[0]["title"] == "Parsed Paper"
    assert chunk_calls[0]["metadata"] == {"chunk_id": "7-p1-c1"}

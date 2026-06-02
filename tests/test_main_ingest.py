from __future__ import annotations

import main as cli_main


def test_ingest_pdf_writes_parsed_paper_fields(monkeypatch) -> None:
    paper_calls = []
    chunk_calls = []

    monkeypatch.setattr(cli_main, "init_db", lambda config_path: None)
    monkeypatch.setattr(
        cli_main,
        "parse_pdf",
        lambda file_path: {
            "file_path": file_path,
            "title": "Parsed Title",
            "authors": ["Alice Smith", "Bob Chen"],
            "year": 2025,
            "abstract": "Parsed abstract.",
            "metadata": {"page_count": 1},
            "pages": [{"page": 1, "text": "Parsed text.", "file_path": file_path}],
        },
    )
    monkeypatch.setattr(
        cli_main,
        "insert_paper",
        lambda **kwargs: paper_calls.append(kwargs) or 7,
    )
    monkeypatch.setattr(
        cli_main,
        "split_pages_to_chunks",
        lambda pages, paper_id, chunk_size, overlap: [
            {"chunk_id": "7-p1-c1", "paper_id": paper_id, "page": 1, "chunk_text": "Parsed text."}
        ],
    )
    monkeypatch.setattr(cli_main, "insert_chunk", lambda **kwargs: chunk_calls.append(kwargs) or 1)
    monkeypatch.setattr(cli_main, "TextEmbeddingModel", lambda config_path: object())
    monkeypatch.setattr(cli_main, "QdrantTextStore", lambda embedding_model, config_path: _FakeQdrantStore())

    result = cli_main.ingest_pdf("paper.pdf", config_path="config.yaml")

    assert result["paper_id"] == 7
    assert paper_calls == [
        {
            "title": "Parsed Title",
            "authors": ["Alice Smith", "Bob Chen"],
            "year": 2025,
            "source_path": "paper.pdf",
            "abstract": "Parsed abstract.",
            "metadata": {"page_count": 1},
            "config_path": "config.yaml",
        }
    ]
    assert chunk_calls[0]["paper_id"] == 7


class _FakeQdrantStore:
    collection_name = "papers_text"

    def create_collection(self) -> None:
        return None

    def upsert_text_chunks(self, chunks) -> int:
        return len(chunks)

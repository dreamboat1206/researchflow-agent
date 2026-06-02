from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.retrieval_agent import RetrievalAgent
from models.llm_client import LLMClient
from storage.sqlite_store import DEFAULT_CONFIG_PATH


INSUFFICIENT_CONTEXT_ANSWER = "未找到足够依据，无法回答该问题。"


class PaperQAAgent:
    def __init__(
        self,
        retrieval_agent: RetrievalAgent | None = None,
        llm_client: LLMClient | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.retrieval_agent = retrieval_agent or RetrievalAgent(config_path=config_path)
        self.llm_client = llm_client or LLMClient(config_path=config_path)

    def answer_question(self, question: str, top_k: int = 5) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            return {"answer": INSUFFICIENT_CONTEXT_ANSWER, "citations": []}
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        chunks = self.retrieval_agent.search_papers(normalized_question, top_k=top_k)
        if not chunks:
            return {"answer": INSUFFICIENT_CONTEXT_ANSWER, "citations": []}

        citations = [_citation_from_chunk(chunk) for chunk in chunks]
        prompt = _build_prompt(normalized_question, chunks)
        answer = self.llm_client.generate(prompt)
        return {
            "answer": _ensure_citation_note(answer, citations),
            "citations": citations,
        }


def _build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or "Unknown paper"
        page = chunk.get("page") or "unknown"
        chunk_id = chunk.get("chunk_id") or "unknown"
        text = chunk.get("chunk_text") or ""
        context_blocks.append(
            f"[{index}] title={title}; page={page}; chunk_id={chunk_id}\n{text}"
        )

    return (
        "Use only the paper chunks below to answer the question. "
        "If the context is insufficient, say you do not have enough evidence. "
        "Include citation markers like [1] in the answer.\n\n"
        f"Question:\n{question}\n\n"
        "Context:\n"
        + "\n\n".join(context_blocks)
    )


def _citation_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": chunk.get("title"),
        "page": chunk.get("page"),
        "chunk_id": chunk.get("chunk_id"),
    }


def _ensure_citation_note(answer: str, citations: list[dict[str, Any]]) -> str:
    if any(f"[{index}]" in answer for index in range(1, len(citations) + 1)):
        return answer
    markers = ", ".join(f"[{index}]" for index in range(1, len(citations) + 1))
    return f"{answer}\n\nCitations: {markers}"

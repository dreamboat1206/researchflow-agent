from __future__ import annotations

from agents.paper_qa_agent import INSUFFICIENT_CONTEXT_ANSWER, PaperQAAgent


class FakeRetrievalAgent:
    def __init__(self, chunks: list[dict[str, object]]):
        self.chunks = chunks

    def search_papers(self, query: str, top_k: int) -> list[dict[str, object]]:
        return self.chunks[:top_k]


class FakeLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "ZoomDet improves UAV object detection by adaptively magnifying object regions. [1]"


def test_answer_question_returns_answer_and_citations() -> None:
    llm_client = FakeLLMClient()
    agent = PaperQAAgent(
        retrieval_agent=FakeRetrievalAgent(
            [
                {
                    "title": "ZoomDet",
                    "page": 2,
                    "chunk_id": "1-p2-c1",
                    "chunk_text": "ZoomDet magnifies object regions for UAV object detection.",
                    "score": 0.9,
                }
            ]
        ),
        llm_client=llm_client,
    )

    result = agent.answer_question("What does ZoomDet do?", top_k=1)

    assert "ZoomDet improves" in result["answer"]
    assert result["citations"] == [
        {"title": "ZoomDet", "page": 2, "chunk_id": "1-p2-c1"}
    ]
    assert "What does ZoomDet do?" in llm_client.prompts[0]
    assert "chunk_id=1-p2-c1" in llm_client.prompts[0]


def test_answer_question_refuses_without_context() -> None:
    agent = PaperQAAgent(
        retrieval_agent=FakeRetrievalAgent([]),
        llm_client=FakeLLMClient(),
    )

    result = agent.answer_question("What is missing?", top_k=3)

    assert result == {"answer": INSUFFICIENT_CONTEXT_ANSWER, "citations": []}

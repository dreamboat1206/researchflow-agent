from __future__ import annotations

from agents.router_agent import RouterAgent, route_query


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_route_query_detects_paper_ingest() -> None:
    result = route_query("请导入这篇 PDF 到知识库")

    assert result == {
        "task_type": "paper_ingest",
        "query": "请导入这篇 PDF 到知识库",
        "need_multimodal": False,
    }


def test_route_query_detects_figure_search() -> None:
    result = route_query("找一下 Transformer 架构图")

    assert result == {
        "task_type": "figure_search",
        "query": "找一下 Transformer 架构图",
        "need_multimodal": True,
    }


def test_route_query_detects_figure_qa() -> None:
    result = route_query("解释一下 Figure 2 说明了什么")

    assert result == {
        "task_type": "figure_qa",
        "query": "解释一下 Figure 2 说明了什么",
        "need_multimodal": True,
    }


def test_route_query_detects_paper_qa() -> None:
    result = route_query("这篇论文的核心贡献是什么")

    assert result == {
        "task_type": "paper_qa",
        "query": "这篇论文的核心贡献是什么",
        "need_multimodal": False,
    }


def test_route_query_detects_organize() -> None:
    result = route_query("帮我整理这些论文并按方法分类")

    assert result == {
        "task_type": "organize",
        "query": "帮我整理这些论文并按方法分类",
        "need_multimodal": False,
    }


def test_uncertain_query_uses_llm_fallback() -> None:
    llm_client = FakeLLMClient("paper_qa")
    agent = RouterAgent(llm_client=llm_client)

    result = agent.route_query("ZoomDet?")

    assert result == {
        "task_type": "paper_qa",
        "query": "ZoomDet?",
        "need_multimodal": False,
    }
    assert "Classify the user request" in llm_client.prompts[0]


def test_unknown_fallback_does_not_route_random_text() -> None:
    agent = RouterAgent(llm_client=FakeLLMClient("unknown"))

    result = agent.route_query("blue coffee calendar")

    assert result == {
        "task_type": "unknown",
        "query": "blue coffee calendar",
        "need_multimodal": False,
    }


def test_invalid_llm_fallback_returns_unknown() -> None:
    agent = RouterAgent(llm_client=FakeLLMClient("definitely-not-a-valid-route"))

    result = agent.route_query("ambiguous request")

    assert result["task_type"] == "unknown"
    assert result["need_multimodal"] is False

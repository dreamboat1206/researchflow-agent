# Codex Project Instruction: ResearchFlow-Agent

你是我的结对编程助手，请帮助我完成 `ResearchFlow-Agent` 项目。

## 项目目标

我要构建一个基于 LangGraph 的本地多模态科研论文知识库 Agent，支持：

1. PDF 论文解析与入库
2. 文本 chunk 切分与向量化
3. 基于 Qdrant 的论文语义检索
4. 论文 RAG 问答，回答必须带论文名和页码引用
5. PDF Figure/Table 自动抽取
6. Figure caption、nearby text、image embedding 的多路图表检索
7. 图表问答
8. LangGraph 多 Agent 工作流
9. Agent trace、检索日志、错误日志
10. evals 自动评测脚本，包括 Recall@K、MRR@K、Citation Coverage、Answer Supported Rate、Latency

## 技术栈

* Python 3.10+
* FastAPI
* Streamlit
* LangGraph
* Qdrant
* SQLite
* PyMuPDF
* sentence-transformers
* CLIP / OpenCLIP
* Pydantic
* pytest
* loguru
* optional: Langfuse

## 编码要求

1. 每次只实现一个清晰的小任务，不要一次性大改。
2. 不要删除已有功能。
3. 新增功能必须有最小测试。
4. 所有工具函数要尽量可单元测试。
5. Agent 层不要写死业务逻辑，底层能力放到 tools、models、storage。
6. LangGraph 节点只负责读取 state、调用工具、写回 state。
7. 所有 RAG 回答必须带 citations。
8. 如果没有检索到可靠上下文，回答应该拒绝编造。
9. 所有数据库路径、模型名、collection 名从 config.yaml 或环境变量读取。
10. 每次修改后请运行相关测试，并告诉我修改了哪些文件。

## 项目目录目标

请尽量保持以下结构：

researchflow_agent/
├── app/
├── api/
├── graph/
├── agents/
├── tools/
├── models/
├── storage/
├── observability/
├── evals/
├── data/
└── tests/

## 工作方式

当我给你一个任务时，请先执行：

1. 阅读当前项目结构
2. 说明你准备修改哪些文件
3. 给出简短实现计划
4. 等我确认或直接按任务规模实现
5. 实现代码
6. 运行测试
7. 总结结果和下一步建议

不要引入过重依赖，不要写无法运行的伪代码。
day1 是main分支
git checkout -b day02-sqlite-store
git checkout -b day03-pdf-parser
git checkout -b day04-text-chunker
git checkout -b day05-qdrant-ingest
git checkout -b day06-text-search
git checkout -b day07-paper-rag
git checkout -b day08-figure-schema
git checkout -b day09-figure-extract
git checkout -b day10-caption-match
git checkout -b day11-figure-gallery
git checkout -b day12-figure-text-search
git checkout -b day13-clip-image-search
git checkout -b day14-fusion-retrieval
git checkout -b day15-langgraph-state
git checkout -b day16-router-agent
git checkout -b day17-paper-qa-graph
git checkout -b day18-figure-search-graph
git checkout -b day19-figure-qa-graph
git checkout -b day20-evaluator-agent
git checkout -b day21-observability
git checkout -b day22-eval-datasets
git checkout -b day23-text-eval
git checkout -b day24-figure-eval
git checkout -b day25-rag-eval
git checkout -b day26-stability
git checkout -b day27-readme-demo
git checkout -b day28-final-polish

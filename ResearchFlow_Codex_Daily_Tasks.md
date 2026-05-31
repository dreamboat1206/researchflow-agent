# ResearchFlow-Agent：每日 Codex 执行指令

> 使用方式：每天复制对应 Day 的指令给 Codex。建议先把本文件放到项目根目录，命名为 `CODEX_DAILY_TASKS.md`。  
> 同时建议把「主控指令」单独复制到项目根目录的 `CODEX_TASK.md`，让 Codex 每次先读取它。

---

## 0. Codex 主控指令：CODEX_TASK.md

```text
# Codex Project Instruction: ResearchFlow-Agent

你是我的结对编程助手，请帮助我完成 ResearchFlow-Agent 项目。

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

- Python 3.10+
- FastAPI
- Streamlit
- LangGraph
- Qdrant
- SQLite
- PyMuPDF
- sentence-transformers
- CLIP / OpenCLIP
- Pydantic
- pytest
- loguru
- optional: Langfuse

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
```

---

## 1. 每日通用模板

```text
请阅读 CODEX_TASK.md 和当前项目代码。

今天的任务是：{填写今天的任务名称}

具体要求：

1. 只完成今天任务，不要实现后续阶段功能。
2. 先检查当前项目结构，说明需要新增或修改的文件。
3. 实现可运行的最小版本。
4. 为核心函数添加 pytest 测试。
5. 如果需要新增依赖，请更新 requirements.txt，并解释原因。
6. 如果需要新增配置，请更新 config.yaml 和 .env.example。
7. 修改完成后运行相关测试。
8. 最后用中文总结：
   - 修改了哪些文件
   - 新增了什么能力
   - 如何运行
   - 测试结果
   - 下一步应该做什么

验收标准：

{填写当天验收标准}
```

---

# 第 1 周：文本 RAG MVP

## Day 1：项目初始化

```text
请阅读 CODEX_TASK.md。

今天任务：初始化 ResearchFlow-Agent 项目骨架。

请完成：
1. 创建推荐目录结构：app、api、graph、agents、tools、models、storage、observability、evals、data、tests。
2. 创建 requirements.txt，包含 FastAPI、Streamlit、Pydantic、PyMuPDF、Qdrant client、sentence-transformers、pytest、loguru 等基础依赖。
3. 创建 config.yaml，包含数据库路径、Qdrant 地址、collection 名、模型名等配置项。
4. 创建 .env.example。
5. 创建 api/server.py，实现一个 /health 接口。
6. 创建 app/streamlit_app.py，实现一个最小 Streamlit 页面。
7. 创建 main.py，提供简单 CLI 占位。
8. 创建 README.md 初版，包含项目简介、技术栈、目录结构、快速开始。
9. 添加 tests/test_health.py 或基础测试。

验收标准：
- uvicorn api.server:app --reload 可以启动。
- streamlit run app/streamlit_app.py 可以启动。
- pytest 可以运行通过。
```

## Day 2：SQLite 元数据存储层

```text
请阅读 CODEX_TASK.md 和当前代码。

今天任务：实现 SQLite 元数据存储层。

请完成：
1. 创建 storage/schema.sql。
2. 创建 papers、chunks、figures、traces 四张表。
3. 实现 storage/sqlite_store.py。
4. 支持 init_db、insert_paper、insert_chunk、insert_figure、list_papers、get_paper。
5. 所有数据库路径从 config.yaml 读取。
6. 添加 tests/test_sqlite_store.py。
7. 更新 README 的数据库部分。

验收标准：
- python main.py init-db 可以初始化数据库。
- pytest tests/test_sqlite_store.py 通过。
- 可以插入并查询 paper/chunk/figure 测试数据。
```

## Day 3：PDF 文本解析

```text
今天任务：实现 PDF 文本解析工具。

请完成：
1. 创建 tools/pdf_parser.py。
2. 使用 PyMuPDF 读取 PDF。
3. 返回每页文本、页码、文件路径。
4. 实现简单 title 提取逻辑：优先取第一页非空较短文本行。
5. 实现 parse_pdf(file_path) 函数。
6. 添加 tests/test_pdf_parser.py，可使用一个临时生成的小 PDF 测试。
7. main.py 增加 parse-pdf 命令。

验收标准：
- parse_pdf 能返回 pages 列表。
- 每页包含 page 和 text。
- 测试通过。
```

## Day 4：文本切分

```text
今天任务：实现论文文本 chunk 切分。

请完成：
1. 创建 tools/text_splitter.py。
2. 实现 split_pages_to_chunks(pages, paper_id, chunk_size, overlap)。
3. 每个 chunk 保留 paper_id、chunk_id、page、chunk_text。
4. 避免空 chunk。
5. 添加 tests/test_text_splitter.py。
6. 在 README 中说明 chunk 策略。

验收标准：
- 输入多页文本能输出 chunks。
- chunk_id 稳定可读。
- 每个 chunk 保留 page。
- pytest 通过。
```

## Day 5：Qdrant 文本向量入库

```text
今天任务：实现 Qdrant 文本向量存储。

请完成：
1. 创建 models/text_embedding.py。
2. 使用 sentence-transformers 封装 TextEmbeddingModel。
3. 创建 storage/qdrant_store.py。
4. 支持 create_collection、upsert_text_chunks、search_text。
5. collection 名和模型名从 config.yaml 读取。
6. 添加 tests/test_qdrant_store.py，能 mock 就 mock，避免测试依赖外部服务。
7. main.py 增加 ingest-pdf 命令，串联 parse_pdf、split_chunks、embedding、qdrant upsert、sqlite insert。

验收标准：
- ingest-pdf 可以处理一个 PDF。
- Qdrant 可用时能写入 papers_text collection。
- search_text 可以返回 top-k chunks。
```

## Day 6：文本检索 API 和 UI

```text
今天任务：实现论文文本搜索接口和 Streamlit 页面。

请完成：
1. 创建 agents/retrieval_agent.py。
2. 实现 search_papers(query, top_k)。
3. 创建 api/routes_paper.py。
4. 实现 POST /search/text。
5. Streamlit 增加“论文搜索”页面。
6. 搜索结果展示 title、page、chunk_text、score。
7. 添加必要测试。

验收标准：
- API 可以根据 query 返回 top-k chunk。
- Streamlit 页面能输入问题并展示结果。
```

## Day 7：RAG 问答 MVP

```text
今天任务：实现论文 RAG 问答 MVP。

请完成：
1. 创建 models/llm_client.py，先支持 OpenAI-compatible API 或本地 mock。
2. 创建 agents/paper_qa_agent.py。
3. 检索 top-k chunks 后构造 prompt。
4. 回答必须包含 citations。
5. 没有检索结果时必须拒绝编造。
6. 创建 POST /qa/paper。
7. Streamlit 增加“论文问答”页面。
8. 添加 tests/test_paper_qa_agent.py。

验收标准：
- 输入问题后能返回 answer 和 citations。
- citations 包含 paper title、page、chunk_id。
- 无检索结果时返回“未找到足够依据”。
```

---

# 第 2 周：图表抽取与图表检索

## Day 8：Figure/Table 元数据设计

```text
今天任务：完善 Figure/Table 元数据设计。

请完成：
1. 检查并完善 SQLite figures 表字段。
2. 定义 figure_id 生成规则：{paper_id}_fig_{page}_{index}。
3. 定义 figure_type 枚举：architecture、pipeline、chart、table、ablation、result、dataset、other。
4. 创建 data/figures 保存目录。
5. 创建 schemas 或 models 中的 FigureRecord 数据结构。
6. 更新 README 中的图表数据结构说明。
7. 添加 tests/test_figure_schema.py。

验收标准：
- figure metadata 可以被创建、校验、写入 SQLite。
- image_path、page、caption、figure_type 字段齐全。
```

## Day 9：PDF Figure 图片抽取

```text
今天任务：实现 PDF Figure 图片抽取。

请完成：
1. 创建 tools/figure_extractor.py。
2. 使用 PyMuPDF 抽取 PDF 中的图片。
3. 保存到 data/figures/{paper_id}/。
4. 过滤宽高过小的图片。
5. 返回 figure_id、paper_id、page、image_path、width、height。
6. 写入 SQLite figures 表。
7. 添加 tests/test_figure_extractor.py。
8. main.py 增加 extract-figures 命令。

验收标准：
- 对真实 PDF 能抽取图片。
- 图片保存路径正确。
- SQLite 中有 figures 记录。
```

## Day 10：Caption 匹配

```text
今天任务：实现 Figure/Table caption 匹配。

请完成：
1. 创建 tools/caption_matcher.py。
2. 从页面文本中匹配 Figure、Fig.、Table、图、表 等 caption。
3. 根据 page 绑定 caption。
4. 为每个 figure 添加 caption 和 nearby_text。
5. 更新 SQLite figures 表。
6. 添加 tests/test_caption_matcher.py。
7. Streamlit Figure Gallery 展示 caption。

验收标准：
- caption_matcher 能识别英文和中文 caption。
- figure 详情中能展示 caption 和 nearby_text。
```

## Day 11：Figure Gallery 页面

```text
今天任务：实现 Figure Gallery 前端页面。

请完成：
1. 在 Streamlit 中增加“Figure Gallery / 图表浏览”页面。
2. 支持按 paper_id 或论文标题筛选。
3. 展示图片缩略图、caption、page、figure_type。
4. 支持点击或展开查看详情。
5. 图表详情展示 image_path、caption、nearby_text、paper_id、page。
6. 如果没有图表，展示友好的空状态提示。
7. 添加必要的前端辅助函数测试，或保证相关数据读取函数有测试。

验收标准：
- 能浏览每篇论文抽出的图表。
- 能看到 caption 和页码。
- Demo 页面观感清晰。
```

## Day 12：基于 caption + nearby_text 的图表检索

```text
今天任务：实现基于 caption + nearby_text 的图表检索。

请完成：
1. 在 storage/qdrant_store.py 中增加 paper_figures_text collection 支持。
2. 对 figure caption + nearby_text 生成 embedding。
3. 实现 upsert_figures_text。
4. 创建 agents/figure_retrieval_agent.py 初版。
5. 实现 search_figures_by_text(query, top_k)。
6. 创建 POST /search/figures。
7. Streamlit 增加“图表搜索”页面。

验收标准：
- 输入“找一下 Transformer 架构图”能返回相关 figure。
- 结果包含 image_path、caption、page、paper_id、score。
```

## Day 13：CLIP / OpenCLIP 图像检索

```text
今天任务：实现 CLIP/OpenCLIP 图像向量检索。

请完成：
1. 创建 models/image_embedding.py。
2. 封装 ImageEmbeddingModel，支持 encode_image 和 encode_text。
3. 创建 paper_figures_image collection。
4. 对 data/figures 中图片生成 image embedding。
5. 实现 search_figures_by_image_text(query, top_k)。
6. 不要破坏已有 caption 检索。
7. 添加 mock 测试或小图片测试。

验收标准：
- 图像 embedding 可以生成。
- 文本 query 可以检索 image embedding。
- 返回 figure_id、image_path、score。
```

## Day 14：图表多路融合检索

```text
今天任务：实现图表多路融合检索。

请完成：
1. 在 agents/figure_retrieval_agent.py 中实现 fusion search。
2. 融合 caption_text_score、image_score、nearby_text_score。
3. 初始权重设置为 0.5、0.3、0.2。
4. 归一化不同检索通道分数。
5. 去重并按 final_score 排序。
6. Streamlit 展示每个通道分数和最终分数。
7. 添加 tests/test_figure_fusion_retrieval.py。

验收标准：
- search_figures(query, mode="fusion") 能返回融合排序结果。
- 每个结果包含 final_score 和 score_breakdown。
```

---

# 第 3 周：LangGraph 多 Agent 化 + 可观测性

## Day 15：LangGraph State

```text
今天任务：定义 LangGraph 全局状态。

请完成：
1. 创建 graph/state.py。
2. 定义 ResearchState TypedDict。
3. 定义 RetrievedChunk、RetrievedFigure、Citation、EvaluationResult。
4. 确保 state 能覆盖 paper_qa、figure_search、figure_qa、paper_ingest。
5. 添加 tests/test_graph_state.py。

验收标准：
- state 类型清晰。
- 后续 graph 节点可以统一读写 state。
```

## Day 16：Router Agent

```text
今天任务：实现 Router Agent。

请完成：
1. 创建 agents/router_agent.py。
2. 支持任务类型：paper_qa、figure_search、figure_qa、paper_ingest、organize、unknown。
3. 先用规则 + LLM fallback 的方式实现。
4. 添加 route_query(user_input)。
5. 添加 tests/test_router_agent.py。
6. 输出 task_type、query、need_multimodal。

验收标准：
- 常见问题能被正确路由。
- 不确定问题返回 unknown 或 paper_qa，不要乱路由。
```

## Day 17：Paper QA Graph

```text
今天任务：实现 LangGraph 论文问答工作流。

请完成：
1. 创建 graph/paper_qa_graph.py。
2. 节点包括 router_node、text_retrieval_node、answer_generation_node、evaluation_node。
3. 每个节点读写 ResearchState。
4. 提供 build_paper_qa_graph()。
5. 提供 invoke_paper_qa(query)。
6. 添加 tests/test_paper_qa_graph.py。

验收标准：
- 输入 query，graph 返回 final_answer 和 citations。
- graph 节点可单独测试。
```

## Day 18：Figure Search Graph

```text
今天任务：实现 LangGraph 图表搜索工作流。

请完成：
1. 创建 graph/figure_search_graph.py。
2. 节点包括 router_node、figure_retrieval_node、format_results_node。
3. 使用已有 figure_retrieval_agent。
4. Streamlit 图表搜索页面改为调用 graph。
5. 添加 tests/test_figure_search_graph.py。

验收标准：
- graph.invoke 返回 retrieved_figures。
- 前端展示不受影响。
```

## Day 19：Figure QA Graph

```text
今天任务：实现图表问答工作流。

请完成：
1. 创建 graph/figure_qa_graph.py。
2. 创建 agents/multimodal_qa_agent.py。
3. 支持根据 figure_id 直接问答。
4. 支持根据 query 先检索 figure 再问答。
5. 回答使用 caption、nearby_text、可选 OCR 文本。
6. 输出必须包含 paper_id、figure_id、page。
7. 添加 tests/test_figure_qa_graph.py。

验收标准：
- 输入 figure_id 和问题，可以返回图表解释。
- 没找到 figure 时明确返回错误信息。
```

## Day 20：Evaluator Agent

```text
今天任务：实现 Evaluator Agent。

请完成：
1. 创建 agents/evaluator_agent.py。
2. 检查回答是否有 citations。
3. 检查 citations 是否来自 retrieved_chunks 或 retrieved_figures。
4. 检查没有检索结果时是否拒答。
5. 输出 EvaluationResult。
6. 集成到 paper_qa_graph 和 figure_qa_graph。
7. 添加 tests/test_evaluator_agent.py。

验收标准：
- 没 citation 的回答会被标记为 invalid。
- 有 citation 且来源正确的回答会被标记为 valid。
```

## Day 21：Agent Trace 和日志记录

```text
今天任务：实现 Agent trace 和日志记录。

请完成：
1. 创建 observability/trace_logger.py。
2. 每次 graph 调用记录 trace_id、task_type、query、retrieved_items、final_answer、latency_ms、success、error_message。
3. 写入 SQLite traces 表。
4. 如果配置了 Langfuse 环境变量，则同步记录到 Langfuse；否则使用本地 SQLite。
5. Streamlit 增加“Trace Viewer”页面。
6. 添加 tests/test_trace_logger.py。

验收标准：
- 每次 QA 或搜索都有 trace。
- Trace Viewer 能看到最近请求。
- 出错时能记录 error_message。
```

---

# 第 4 周：评测、优化、项目包装

## Day 22：构建评测集

```text
今天任务：构建 eval 数据集样例和标注格式。

请完成：
1. 创建 evals/datasets/qa_eval.example.jsonl。
2. 创建 evals/datasets/figure_eval.example.jsonl。
3. 创建 evals/datasets/classification_eval.example.jsonl。
4. 为每类数据提供 3-5 条样例。
5. 在 README 或 evals/README.md 中说明如何扩展到 200 条 QA 和 100 条 figure query。
6. 添加 JSONL 格式校验脚本 evals/validate_datasets.py。
7. 添加 tests/test_eval_dataset_format.py。

验收标准：
- eval jsonl 格式固定。
- 每条 QA 至少包含 query、expected_paper、expected_pages、expected_keywords。
- 每条 figure query 至少包含 query、expected_figure_id 或 expected_paper/expected_page。
```

## Day 23：文本检索评测

```text
今天任务：实现文本检索评测脚本。

请完成：
1. 创建 evals/metrics.py。
2. 实现 recall_at_k、mrr_at_k、page_hit_rate。
3. 创建 evals/run_text_retrieval_eval.py。
4. 读取 qa_eval jsonl。
5. 调用现有 retrieval agent。
6. 输出 Markdown 表格到 evals/results/text_retrieval_results.md。
7. 添加 tests/test_metrics.py。

验收标准：
- 可以运行 python evals/run_text_retrieval_eval.py。
- 输出 Recall@5、Recall@10、MRR@10、Page Hit Rate、Latency。
```

## Day 24：图表检索评测

```text
今天任务：实现图表检索评测脚本。

请完成：
1. 创建 evals/run_figure_retrieval_eval.py。
2. 读取 figure_eval jsonl。
3. 对比 caption-only、image-only、fusion 三种模式。
4. 输出 Figure Recall@5、Recall@10、MRR@10、Latency。
5. 生成 evals/results/figure_retrieval_results.md。
6. 添加相关测试，至少测试 metrics 计算逻辑。

验收标准：
- 可以运行图表检索评测。
- 可以看到三种检索模式对比表。
```

## Day 25：RAG 回答质量评测

```text
今天任务：实现 RAG 回答质量评测。

请完成：
1. 创建 evals/run_rag_eval.py。
2. 统计 Citation Coverage。
3. 统计 Answer Supported Rate。
4. 统计 Refusal Accuracy。
5. 统计 Hallucination Rate 的规则版估计。
6. 输出 evals/results/rag_eval_results.md。
7. README 中加入评测结果占位表。

验收标准：
- 可以运行 RAG eval。
- 输出引用覆盖率、上下文支撑率、拒答准确率、幻觉率。
```

## Day 26：性能优化与错误处理

```text
今天任务：性能优化与错误处理。

请完成：
1. 为 embedding 增加批处理。
2. 为 Qdrant upsert 增加批量写入。
3. 为 PDF 解析和图表抽取增加异常捕获。
4. 为外部模型调用增加超时和重试。
5. 检索为空时统一返回拒答，不允许编造。
6. 为长 PDF 处理增加进度日志。
7. 统计 workflow success rate 和 average latency。
8. 添加 tests/test_error_handling.py 或更新相关测试。

验收标准：
- 300 次模拟请求中，工作流不因单条错误中断整体服务。
- 平均 QA 延迟和图表搜索延迟可被 trace_logger 记录。
- 空检索结果能稳定拒答。
```

## Day 27：README 和项目展示材料

```text
今天任务：完善 README 和项目展示材料。

请完成：
1. 完善 README.md。
2. 加入项目简介、核心功能、技术架构、目录结构、快速开始、Demo 示例、评测结果、项目亮点。
3. 用 Mermaid 画系统架构图。
4. 用 Mermaid 画 LangGraph 工作流图。
5. 加入 evals/results 中的结果表格。
6. 加入简历写法参考。
7. 检查所有命令是否准确。

验收标准：
- README 能让别人理解并运行项目。
- README 有架构图、工作流图、评测结果。
```

## Day 28：最终整理和简历材料

```text
今天任务：最终整理和简历材料。

请完成：
1. 清理无用代码、TODO 和调试输出。
2. 确认 .gitignore 不会上传 .env、data、数据库文件、模型缓存。
3. 补充 README 的最终运行命令。
4. 汇总 evals/results，生成 evals/results/summary.md。
5. 创建 docs/resume_project_description.md，包含长版和短版简历描述。
6. 创建 docs/interview_talk_track.md，包含 2 分钟项目讲解稿。
7. 运行 pytest，记录最终测试结果。

验收标准：
- 本地能一键启动 API 和 Streamlit。
- README、eval summary、简历描述、面试讲解稿完整。
- pytest 通过或明确记录无法通过的原因。
```

---

# 附录：Git + Codex 安全工作流

每天开始前：

```bash
git checkout main
git pull
git checkout -b dayXX-task-name
```

让 Codex 改代码前：

```bash
git status
```

Codex 改完后：

```bash
git diff
pytest
git add .
git commit -m "feat: dayXX task summary"
```

测试通过后合并：

```bash
git checkout main
git merge dayXX-task-name
git push
```

如果 Codex 改坏了：

```bash
git restore .
git clean -fd
```

注意：`git clean -fd` 会删除未被 Git 跟踪的新文件，执行前先确认。

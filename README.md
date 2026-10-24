# 计算机网络课程 RAG 知识问答系统

本项目将课程教材、PPT、Markdown 和 TXT 资料解析为可追溯的文本块，提供三种可比较的问答模式：直接调用本地模型、基础 RAG、RAG + CrossEncoder 重排序。RAG 回答附带经校验的来源页码和 chunk ID；资料不足时会明确拒答。

## 目录

- `data/documents/`：本地原始教材和课件。
- `data/processed/`：清洗后的分块预览（运行入库后生成）。
- `data/chroma/`：Chroma 持久化向量库。
- `data/evals/`：评测题、运行结果与汇总 CSV。
- `notebooks/`：从原型阶段保留的实验记录。
- `docs/`：项目执行计划和原型代码梳理。

## 安装

```powershell
cd D:\ds_rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

确保 Ollama 已启动，并已拉取 `.env` 中指定的模型，例如：

```powershell
ollama pull deepseek-r1:7b
```

## 常用命令

```powershell
# 建库；--force 会重建同一文档内容哈希对应的向量
rag ingest "data/documents/test.pdf"

# 查看指定页面的已清洗文本
rag inspect --document "data/documents/test.pdf" --page 1

# 三种问答模式
rag ask "TCP 为什么需要三次握手？" --mode llm_direct
rag ask "TCP 为什么需要三次握手？" --mode rag
rag ask "TCP 为什么需要三次握手？" --mode rag_rerank --debug

# 批量评测；结果写入 data/evals/results/
rag evaluate data/evals/questions.sample.jsonl

# 图形界面
streamlit run src/rag/app.py
```

`ingest` 会在 `data/processed/<document_id>.chunks.jsonl` 写入可人工抽查的分块预览。文件内容的 SHA-256 前 16 位是文档 ID，因此资料改版后会被识别为不同版本。

## 验证

```powershell
ruff check src tests
pytest
```

模型、Chroma 和向量库均在本机运行。不要将含原始课程资料、`.env` 或生成的评测结果提交到公开仓库。

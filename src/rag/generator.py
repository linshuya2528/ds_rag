"""Ollama client and evidence-constrained answer orchestration."""

from __future__ import annotations

import time
from typing import Any

from .config import Settings, get_settings
from .reranker import Reranker
from .retriever import Retriever
from .schemas import Answer, Citation, RetrievedChunk
from .validator import parse_json_response, validate_payload

INSUFFICIENT_EVIDENCE = "知识库中未检索到足够依据，无法可靠回答该问题。"


def build_prompt(question: str, evidence: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[chunk_id={item.chunk.id}; source={item.chunk.source}; "
        f"pages={item.chunk.page_start}-{item.chunk.page_end}]\n{item.chunk.text}"
        for item in evidence
    )
    return f"""你是严谨的课程教材问答助手。只能使用下面的参考资料回答。
若资料无法支持回答，必须明确拒答，不能编造事实或引用。
只返回一个 JSON 对象，不能使用 Markdown。格式为：
{{"answer":"简洁回答", "claims":["可选要点"], "citations":[{{"chunk_id":"资料中实际出现的 ID"}}], "insufficient_evidence":false}}

问题：{question}

参考资料：
{context}"""


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests to call Ollama.") from exc
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json()["response"])
        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to reach Ollama. Start Ollama and confirm OLLAMA_BASE_URL/model settings."
            ) from exc
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Ollama returned an unexpected response.") from exc


def _citations_from_payload(payload: dict[str, Any], evidence: list[RetrievedChunk]) -> list[Citation]:
    by_id = {item.chunk.id: item.chunk for item in evidence}
    result: list[Citation] = []
    for item in payload.get("citations", []):
        chunk = by_id.get(item.get("chunk_id")) if isinstance(item, dict) else None
        if chunk and all(existing.chunk_id != chunk.id for existing in result):
            result.append(Citation(chunk.id, chunk.source, chunk.page_start, chunk.page_end))
    return result


def ask(
    question: str,
    mode: str = "rag_rerank",
    settings: Settings | None = None,
    debug: bool = False,
) -> Answer:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if mode not in {"llm_direct", "rag", "rag_rerank"}:
        raise ValueError("mode must be llm_direct, rag or rag_rerank")
    settings = settings or get_settings()
    started = time.perf_counter()
    evidence: list[RetrievedChunk] = []
    if mode != "llm_direct":
        candidates = Retriever(settings).retrieve(question)
        evidence = candidates if mode == "rag" else Reranker(settings).rerank(question, candidates)
        if not evidence:
            return Answer(
                mode=mode,
                question=question,
                answer=INSUFFICIENT_EVIDENCE,
                retrieved_chunks=[],
                latency_ms=int((time.perf_counter() - started) * 1000),
                insufficient_evidence=True,
            )
    prompt = build_prompt(question, evidence) if evidence else (
        "请只返回 JSON："
        '{"answer":"回答", "citations":[], "insufficient_evidence":false}\\n问题：' + question
    )
    raw = OllamaClient(settings).generate(prompt)
    payload, issues = parse_json_response(raw)
    validation = {"passed": False, "issues": issues}
    if payload is not None:
        validation = validate_payload(payload, evidence)
    answer_text = payload.get("answer", "未能解析模型回答。") if payload else "未能解析模型回答。"
    insufficient = bool(payload.get("insufficient_evidence", False)) if payload else True
    return Answer(
        mode=mode,
        question=question,
        answer=answer_text,
        citations=_citations_from_payload(payload or {}, evidence),
        retrieved_chunks=evidence if debug else evidence,
        latency_ms=int((time.perf_counter() - started) * 1000),
        validation=validation,
        raw_response=raw if debug else None,
        insufficient_evidence=insufficient,
    )

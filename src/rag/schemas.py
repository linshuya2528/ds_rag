"""Stable data contracts shared by ingestion, retrieval, generation and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    document_id: str
    source: str
    page: int
    text: str
    chapter: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    char_count: int
    chapter: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "chapter": self.chapter or "",
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    vector_rank: int
    vector_distance: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    page_start: int
    page_end: int


@dataclass
class Answer:
    mode: str
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: int = 0
    validation: dict[str, Any] = field(default_factory=lambda: {"passed": True, "issues": []})
    raw_response: str | None = None
    insufficient_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

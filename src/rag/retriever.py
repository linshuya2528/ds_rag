"""Vector retrieval, metadata filters and duplicate suppression."""

from __future__ import annotations

from typing import Any

from .config import Settings
from .indexer import Indexer
from .schemas import Chunk, RetrievedChunk


class Retriever:
    def __init__(self, settings: Settings, indexer: Indexer | None = None):
        self.settings = settings
        self.indexer = indexer or Indexer(settings)

    def retrieve(
        self,
        question: str,
        candidate_count: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        count = candidate_count or self.settings.candidate_count
        if self.indexer.collection.count() == 0:
            return []
        embedding = self.indexer.embedder.encode(question).tolist()
        kwargs: dict[str, Any] = {"query_embeddings": [embedding], "n_results": count}
        if metadata_filter:
            kwargs["where"] = metadata_filter
        result = self.indexer.collection.query(**kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        seen: set[str] = set()
        retrieved: list[RetrievedChunk] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            fingerprint = " ".join(text.split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            chunk = Chunk(
                id=chunk_id,
                document_id=metadata["document_id"],
                source=metadata["source"],
                page_start=int(metadata["page_start"]),
                page_end=int(metadata["page_end"]),
                chunk_index=int(metadata["chunk_index"]),
                text=text,
                char_count=len(text),
                chapter=metadata.get("chapter") or None,
            )
            retrieved.append(RetrievedChunk(chunk, rank, float(distance)))
        return retrieved

"""Embedding, durable Chroma storage and idempotent document ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunker import build_chunks
from .cleaners import clean_pages
from .config import Settings
from .loaders import load_document


@dataclass(frozen=True)
class IngestSummary:
    document_id: str
    source: str
    pages: int
    chunks: int
    skipped: bool = False


class Indexer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_directories()
        self._embedder = None
        self._collection = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            self._collection = client.get_or_create_collection(self.settings.collection_name)
        return self._collection

    def has_document(self, document_id: str) -> bool:
        return bool(self.collection.get(where={"document_id": document_id}, limit=1).get("ids"))

    def ingest(self, path: str | Path, force: bool = False) -> IngestSummary:
        pages = clean_pages(load_document(path))
        if not pages:
            raise ValueError(f"No extractable text found in {path}")
        document_id = pages[0].document_id
        if self.has_document(document_id) and not force:
            return IngestSummary(document_id, pages[0].source, len(pages), 0, skipped=True)
        if force:
            self.collection.delete(where={"document_id": document_id})
        chunks = build_chunks(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        self._write_preview(document_id, chunks)
        embeddings = self.embedder.encode([chunk.text for chunk in chunks], show_progress_bar=True)
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[chunk.metadata() for chunk in chunks],
        )
        return IngestSummary(document_id, pages[0].source, len(pages), len(chunks))

    def _write_preview(self, document_id: str, chunks) -> None:
        preview = self.settings.processed_dir / f"{document_id}.chunks.jsonl"
        with preview.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

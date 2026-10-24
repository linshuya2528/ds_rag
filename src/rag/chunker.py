"""Paragraph-first chunking with overlap only for oversized paragraphs."""

from __future__ import annotations

from collections.abc import Iterable

from .cleaners import paragraphs
from .schemas import Chunk, Page


def split_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    parts: list[str] = []
    start = 0
    while start < len(text):
        part = text[start : start + chunk_size].strip()
        if part:
            parts.append(part)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return parts


def chunk_paragraphs(items: Iterable[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in items:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_by_chars(paragraph, chunk_size, overlap))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= chunk_size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def build_chunks(pages: Iterable[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for index, text in enumerate(chunk_paragraphs(paragraphs(page.text), chunk_size, overlap), start=1):
            chunks.append(
                Chunk(
                    id=f"{page.document_id}-p{page.page:03d}-c{index:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page_start=page.page,
                    page_end=page.page,
                    chunk_index=index,
                    text=text,
                    char_count=len(text),
                    chapter=page.chapter,
                )
            )
    return chunks

"""Conservative cleanup that preserves the wording needed for citations."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .schemas import Page


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_repeated_margins(pages: Iterable[Page], min_ratio: float = 0.3) -> set[str]:
    """Detect repeated first/last lines, commonly PDF headers and footers."""
    pages = list(pages)
    if not pages:
        return set()
    candidates: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        # A short page can make a body line appear in both slices; count it once per page.
        for line in set(lines[:2] + lines[-2:]):
            normalized = re.sub(r"\d+", "#", line)
            if len(normalized) >= 3:
                candidates[normalized] += 1
    threshold = max(2, int(len(pages) * min_ratio + 0.999))
    return {line for line, count in candidates.items() if count >= threshold}


def clean_pages(pages: Iterable[Page]) -> list[Page]:
    original = list(pages)
    margins = detect_repeated_margins(original)
    cleaned: list[Page] = []
    for page in original:
        lines = [
            line for line in page.text.splitlines()
            if re.sub(r"\d+", "#", line.strip()) not in margins
        ]
        text = normalize_text("\n".join(lines))
        if text:
            cleaned.append(Page(page.document_id, page.source, page.page, text, page.chapter))
    return cleaned


def paragraphs(text: str) -> list[str]:
    """Split on blank lines, merging wrapped lines inside each paragraph."""
    return [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]

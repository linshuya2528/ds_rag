"""Validate model JSON and make sure every citation belongs to supplied evidence."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import RetrievedChunk


REQUIRED_KEYS = {"answer", "citations", "insufficient_evidence"}


def parse_json_response(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    candidate = raw.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None, ["Model response is not valid JSON."]
    if not isinstance(value, dict):
        return None, ["Model JSON must be an object."]
    return value, []


def validate_payload(payload: dict[str, Any], evidence: list[RetrievedChunk]) -> dict[str, Any]:
    issues: list[str] = []
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        issues.append(f"Missing required keys: {', '.join(sorted(missing))}.")
    if not isinstance(payload.get("answer", ""), str):
        issues.append("answer must be a string.")
    if not isinstance(payload.get("insufficient_evidence", False), bool):
        issues.append("insufficient_evidence must be a boolean.")
    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        issues.append("citations must be a list.")
        citations = []
    evidence_ids = {item.chunk.id for item in evidence}
    for citation in citations:
        if not isinstance(citation, dict) or not isinstance(citation.get("chunk_id"), str):
            issues.append("Each citation must contain a string chunk_id.")
        elif citation["chunk_id"] not in evidence_ids:
            issues.append(f"Citation {citation['chunk_id']} was not supplied as evidence.")
    if not payload.get("insufficient_evidence", False) and not citations and evidence:
        issues.append("An evidence-grounded answer must cite at least one supplied chunk.")
    return {"passed": not issues, "issues": issues}

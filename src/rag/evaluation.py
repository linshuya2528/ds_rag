"""Batch evaluation, retrieval metrics and CSV/JSONL exports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .generator import ask


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieval_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for record in records:
        expected_pages = set(record.get("expected_pages", []))
        ranks = [
            item["vector_rank"]
            for item in record.get("retrieved_chunks", [])
            if item["chunk"]["page_start"] in expected_pages
        ]
        recalls.append(float(bool(ranks)))
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
    return {"recall_at_k": sum(recalls) / len(recalls), "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)}


def run_evaluation(
    questions_path: str | Path, settings: Settings, modes: tuple[str, ...] = ("llm_direct", "rag", "rag_rerank")
) -> tuple[Path, Path]:
    questions = load_questions(questions_path)
    output_dir = settings.evals_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = output_dir / f"evaluation-{stamp}.jsonl"
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            for mode in modes:
                result = ask(question["question"], mode=mode, settings=settings, debug=True).to_dict()
                row = {"evaluation": question, "result": result}
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = output_dir / f"evaluation-{stamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "mode", "latency_ms", "validated", "insufficient_evidence"])
        writer.writeheader()
        for row in rows:
            result = row["result"]
            writer.writerow({
                "id": row["evaluation"]["id"],
                "mode": result["mode"],
                "latency_ms": result["latency_ms"],
                "validated": result["validation"]["passed"],
                "insufficient_evidence": result["insufficient_evidence"],
            })
    return jsonl_path, csv_path

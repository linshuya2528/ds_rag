"""Command line entry point: ingest, inspect, ask and evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cleaners import clean_pages
from .config import get_settings
from .evaluation import run_evaluation
from .generator import ask
from .indexer import Indexer
from .loaders import load_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Local course-material RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="parse and index a document")
    ingest.add_argument("document", type=Path)
    ingest.add_argument("--force", action="store_true", help="replace vectors for this document hash")
    inspect = subparsers.add_parser("inspect", help="print cleaned text for one page")
    inspect.add_argument("--document", type=Path, required=True)
    inspect.add_argument("--page", type=int, required=True)
    ask_parser = subparsers.add_parser("ask", help="ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=("llm_direct", "rag", "rag_rerank"), default="rag_rerank")
    ask_parser.add_argument("--debug", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="run JSONL evaluation questions")
    evaluate.add_argument("questions", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    if args.command == "ingest":
        summary = Indexer(settings).ingest(args.document, force=args.force)
        message = "skipped existing document" if summary.skipped else "indexed"
        print(f"{message}: {summary.source}; pages={summary.pages}; chunks={summary.chunks}; id={summary.document_id}")
    elif args.command == "inspect":
        page = next((item for item in clean_pages(load_document(args.document)) if item.page == args.page), None)
        if page is None:
            raise SystemExit(f"No extracted text on page {args.page}.")
        print(page.text)
    elif args.command == "ask":
        print(json.dumps(ask(args.question, args.mode, settings, args.debug).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        jsonl_path, csv_path = run_evaluation(args.questions, settings)
        print(f"results: {jsonl_path}\nsummary: {csv_path}")


if __name__ == "__main__":
    main()

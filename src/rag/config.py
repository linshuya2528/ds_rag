"""Configuration loaded from environment variables and .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps pure utility tests importable before installation.
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[2]


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    ollama_model: str = "deepseek-r1:7b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    collection_name: str = "computer_network"
    chunk_size: int = 500
    chunk_overlap: int = 80
    candidate_count: int = 10
    top_k: int = 3
    timeout_seconds: int = 120
    temperature: float = 0.0
    max_tokens: int = 800

    @property
    def documents_dir(self) -> Path:
        return self.root_dir / "data" / "documents"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.root_dir / "data" / "chroma"

    @property
    def evals_dir(self) -> Path:
        return self.root_dir / "data" / "evals"

    def ensure_directories(self) -> None:
        for directory in (self.documents_dir, self.processed_dir, self.chroma_dir, self.evals_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings(root_dir: Path | None = None) -> Settings:
    """Read settings once per application entry point."""
    resolved_root = root_dir or ROOT_DIR
    if load_dotenv is not None:
        load_dotenv(resolved_root / ".env")
    settings = Settings(
        root_dir=resolved_root,
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        ollama_model=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        collection_name=os.getenv("CHROMA_COLLECTION", "computer_network"),
        chunk_size=_int_env("CHUNK_SIZE", 500),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 80),
        candidate_count=_int_env("CANDIDATE_COUNT", 10),
        top_k=_int_env("TOP_K", 3),
        timeout_seconds=_int_env("OLLAMA_TIMEOUT_SECONDS", 120),
        temperature=_float_env("TEMPERATURE", 0.0),
        max_tokens=_int_env("MAX_TOKENS", 800),
    )
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
    return settings

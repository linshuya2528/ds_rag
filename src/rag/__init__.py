"""Local, evidence-grounded RAG for course materials."""

from .generator import ask
from .schemas import Answer, Chunk, Page, RetrievedChunk

__all__ = ["Answer", "Chunk", "Page", "RetrievedChunk", "ask"]

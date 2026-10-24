"""CrossEncoder second-stage reranking."""

from __future__ import annotations

from .config import Settings
from .schemas import RetrievedChunk


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.reranker_model)
        return self._model

    def rerank(
        self, question: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.model.predict([[question, item.chunk.text] for item in candidates])
        ranked = [
            RetrievedChunk(item.chunk, item.vector_rank, item.vector_distance, float(score))
            for item, score in zip(candidates, scores)
        ]
        ranked.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)
        return ranked[: top_k or self.settings.top_k]

"""
rag_core.embedders.sentence_transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Embedder backed by the ``sentence-transformers`` library.
"""

from __future__ import annotations

from typing import List

from ..config import EmbedderConfig
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder that uses a SentenceTransformer model.

    Args:
        config: An ``EmbedderConfig`` instance specifying the model name.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = config.model
        self._model = SentenceTransformer(self.model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    # ---- BaseEmbedder interface ----

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self._dimensions

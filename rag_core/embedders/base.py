"""
rag_core.embedders.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all embedders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Interface that every embedder must implement.

    An embedder converts text into a fixed-length vector (list of floats)
    that captures the semantic meaning of the text.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the text embedding.
        """

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once.

        The default implementation calls ``embed()`` in a loop.
        Subclasses may override this for batch-optimised performance.

        Args:
            texts: A list of input texts.

        Returns:
            A list of embeddings, one per input text.
        """
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings produced by this model.

        Subclasses should override this if the value is known at init time.
        The default implementation embeds a probe string to determine the size.
        """
        probe = self.embed("dimension probe")
        return len(probe)

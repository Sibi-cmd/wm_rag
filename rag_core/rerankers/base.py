"""
rag_core.rerankers.base
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all rerankers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import SearchResult


class BaseReranker(ABC):
    """Interface that every reranker must implement.

    A reranker re-scores search results to improve relevance
    beyond raw vector similarity.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        """Re-score and re-order search results.

        Args:
            query: The original user query.
            results: Initial search results from the vector store.
            top_k: Number of top results to keep after reranking.

        Returns:
            A reranked list of ``SearchResult`` objects (best first).
        """

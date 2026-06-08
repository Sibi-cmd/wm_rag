"""
rag_core.vector_stores.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all vector stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import SearchResult


class BaseVectorStore(ABC):
    """Interface that every vector store must implement.

    A vector store persists embedding vectors alongside metadata and
    supports similarity search.
    """

    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors in the store.

        Each item in ``vectors`` must have the keys:
            - ``id``       (str):  Unique identifier.
            - ``values``   (list[float]): The embedding vector.
            - ``metadata`` (dict): Arbitrary metadata to store alongside.

        Args:
            vectors: A list of vector records to upsert.
        """

    @abstractmethod
    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Find the most similar vectors.

        Args:
            vector: The query embedding.
            top_k: Number of results to return.
            filters: Optional metadata filters (provider-specific).

        Returns:
            A list of ``SearchResult`` objects ordered by descending similarity.
        """

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
    ) -> None:
        """Delete vectors from the store.

        Args:
            ids: Specific vector IDs to delete.
            delete_all: If ``True``, delete every vector in the store.
        """

    def persist(self) -> None:
        """Persist the index to disk (for local stores like FAISS).

        Cloud-backed stores (Pinecone, Weaviate) can leave this as a no-op.
        """

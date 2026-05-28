"""
rag_core.vector_stores.pinecone_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vector store backed by Pinecone (cloud).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import VectorStoreConfig
from ..types import SearchResult
from .base import BaseVectorStore


class PineconeStore(BaseVectorStore):
    """Pinecone-backed vector store.

    Args:
        config: A ``VectorStoreConfig`` with ``provider="pinecone"``.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        from pinecone import Pinecone

        api_key = config.api_key
        if not api_key:
            raise ValueError(
                f"Pinecone API key not found. "
                f"Set the environment variable '{config.api_key_env}'."
            )

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(config.index_name)
        self._index_name = config.index_name

    # ---- BaseVectorStore interface ----

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch)

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query_kwargs: Dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filters:
            query_kwargs["filter"] = filters

        response = self._index.query(**query_kwargs)

        results: List[SearchResult] = []
        if hasattr(response, "matches") and response.matches:
            for match in response.matches:
                metadata = match.metadata or {}
                results.append(
                    SearchResult(
                        chunk_id=match.id,
                        text=metadata.get("text", ""),
                        score=float(match.score),
                        metadata=metadata,
                    )
                )
        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
    ) -> None:
        if delete_all:
            self._index.delete(delete_all=True)
        elif ids:
            self._index.delete(ids=ids)

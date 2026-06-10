"""
rag_core.vector_stores.qdrant_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vector store backed by Qdrant.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from ..config import VectorStoreConfig
from ..types import SearchResult
from .base import BaseVectorStore

logger = logging.getLogger("rag_core.vector_stores.qdrant")


class QdrantStore(BaseVectorStore):
    """Qdrant-backed vector store.

    Args:
        config: A ``VectorStoreConfig`` with ``provider="qdrant"``.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Resolve overrides from config extra settings
        qdrant_url = config.extra.get("url", qdrant_url)
        qdrant_api_key = config.extra.get("api_key", qdrant_api_key)

        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._collection_name = config.index_name

    # ---- BaseVectorStore interface ----

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        points = []
        for vec in vectors:
            vec_id: str = vec["id"]
            # Convert string ID to a valid UUID format (Qdrant requirement)
            try:
                uuid.UUID(vec_id)
                point_id = vec_id
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, vec_id))

            values = vec["values"]
            metadata = vec.get("metadata", {})

            # Support both flat and nested metadata retrieval styles
            payload = {
                "metadata": metadata,
                **metadata
            }

            points.append(PointStruct(id=point_id, vector=values, payload=payload))

        if points:
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self._client.upsert(collection_name=self._collection_name, points=batch)

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            query_filter = Filter(must=conditions)

        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
        except Exception as exc:
            logger.warning(f"Filtered search failed, retrying without filters: {exc}")
            # Fallback if filters fail (e.g. index fields not mapped yet)
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=top_k,
                with_payload=True
            )

        points = getattr(results, "points", results)

        # Fallback query without filter if no matches found with filter
        if not points and query_filter:
            try:
                results = self._client.query_points(
                    collection_name=self._collection_name,
                    query=vector,
                    limit=top_k,
                    with_payload=True
                )
                points = getattr(results, "points", results)
            except Exception:
                points = []

        search_results: List[SearchResult] = []
        for item in points:
            payload = getattr(item, "payload", {}) or {}
            metadata = payload.get("metadata") if "metadata" in payload else payload
            search_results.append(
                SearchResult(
                    chunk_id=str(getattr(item, "id", "")),
                    text=metadata.get("text", "") if isinstance(metadata, dict) else "",
                    score=float(getattr(item, "score", 0.0)),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return search_results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
    ) -> None:
        if delete_all:
            # Use FilterSelector with an empty filter to match all points
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=FilterSelector(filter=Filter()),
            )
            logger.info(f"Deleted all points from collection '{self._collection_name}'")
        elif ids:
            point_ids = []
            for item in ids:
                try:
                    uuid.UUID(item)
                    point_ids.append(item)
                except ValueError:
                    point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, item)))
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points from collection '{self._collection_name}'")


"""
rag_core.vector_stores.faiss_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vector store backed by FAISS (local, no cloud needed).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import VectorStoreConfig
from ..types import SearchResult
from .base import BaseVectorStore


class FAISSStore(BaseVectorStore):
    """Local FAISS-backed vector store.

    Stores vectors in a FAISS index on disk.  Good for development,
    testing, and small-to-medium datasets that don't need cloud infra.

    Args:
        config: A ``VectorStoreConfig`` with ``provider="faiss"``.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        import numpy as np

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for the FAISS vector store. "
                "Install it with: pip install faiss-cpu"
            )

        self._faiss = faiss
        self._np = np
        self._dimensions = config.dimensions
        self._index_path = Path(config.index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)

        self._faiss_path = self._index_path / "index.faiss"
        self._meta_path = self._index_path / "metadata.json"

        # Try to load existing index, otherwise create a new one
        if self._faiss_path.exists():
            self._index = faiss.read_index(str(self._faiss_path))
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                self._metadata: Dict[str, Dict[str, Any]] = json.load(fh)
            self._id_list: List[str] = list(self._metadata.keys())
        else:
            self._index = faiss.IndexFlatIP(self._dimensions)  # Inner-product (cosine after normalisation)
            self._metadata = {}
            self._id_list = []

    # ---- BaseVectorStore interface ----

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        np = self._np
        for vec in vectors:
            vec_id: str = vec["id"]
            values = np.array([vec["values"]], dtype=np.float32)

            # L2-normalise so inner-product ≈ cosine similarity
            norm = np.linalg.norm(values, axis=1, keepdims=True)
            if norm > 0:
                values = values / norm

            # If ID already exists, we cannot easily update FAISS in-place,
            # so we just append (simple strategy; fine for moderate scale).
            self._index.add(values)
            self._id_list.append(vec_id)
            self._metadata[vec_id] = vec.get("metadata", {})

        self.persist()

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        np = self._np

        if self._index.ntotal == 0:
            return []

        query = np.array([vector], dtype=np.float32)
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        if norm > 0:
            query = query / norm

        # Search more than top_k if we need to post-filter
        search_k = min(top_k * 3, self._index.ntotal) if filters else min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, search_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            chunk_id = self._id_list[idx]
            meta = self._metadata.get(chunk_id, {})

            # Apply metadata filters
            if filters:
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=meta.get("text", ""),
                    score=float(score),
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
    ) -> None:
        if delete_all:
            self._index = self._faiss.IndexFlatIP(self._dimensions)
            self._metadata = {}
            self._id_list = []
            self.persist()
        elif ids:
            # FAISS IndexFlatIP doesn't support selective deletion.
            # Rebuild the index without the deleted IDs.
            np = self._np
            keep_indices = [i for i, vid in enumerate(self._id_list) if vid not in set(ids)]
            if not keep_indices:
                self.delete(delete_all=True)
                return

            all_vectors = self._faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._dimensions
            )
            all_vectors = np.array(all_vectors).reshape(self._index.ntotal, self._dimensions)
            kept = all_vectors[keep_indices]

            new_index = self._faiss.IndexFlatIP(self._dimensions)
            new_index.add(kept.astype(np.float32))

            new_id_list = [self._id_list[i] for i in keep_indices]
            for vid in ids:
                self._metadata.pop(vid, None)

            self._index = new_index
            self._id_list = new_id_list
            self.persist()

    def persist(self) -> None:
        self._faiss.write_index(self._index, str(self._faiss_path))
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False)

"""
rag_core.rerankers.warehouse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Custom reranker that fuses semantic score, word overlap, section matching, and severity boost.
"""

from __future__ import annotations

import re
from typing import List

from ..types import SearchResult
from .base import BaseReranker


class WarehouseReranker(BaseReranker):
    """Custom reranker tailored for warehouse query and manual context.

    Uses a weighted blend of semantic score, word overlap, section keyword matching,
    and high-severity metadata boost.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_rerank_score(query: str, result: SearchResult) -> float:
        """Calculates custom score blending semantic similarity, lexical overlap, and metadata."""
        sim_score = result.score
        metadata = result.metadata or {}
        text = result.text.lower()
        query_lower = (query or "").lower()

        # 1. Word overlap calculation
        query_words = set(re.findall(r"\w+", query_lower))
        text_words = set(re.findall(r"\w+", text))
        overlap = len(query_words & text_words) / max(len(query_words), 1)

        # 2. Section alignment boost (e.g. section title appears in search query)
        section = metadata.get("section", "").lower()
        section_match = 1.0 if section in query_lower and section else 0.0

        # 3. Severity boost
        severity = metadata.get("severity", "low").lower()
        severity_boost = 1.0 if severity == "high" else (0.5 if severity == "medium" else 0.0)

        # Weighted calculation
        return (0.6 * sim_score) + (0.2 * overlap) + (0.1 * section_match) + (0.1 * severity_boost)

    # ---- BaseReranker interface ----

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        if not results:
            return []

        # Filter out generic navigational/metadata headings
        filtered = [
            r for r in results
            if r.metadata.get("section", "").upper() not in ["CONTENTS", "INDEX", "PREFACE"]
        ]
        target_results = filtered if filtered else results

        scored_results: List[SearchResult] = []
        for r in target_results:
            new_score = self.calculate_rerank_score(query, r)
            scored_results.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=new_score,
                    metadata=r.metadata,
                )
            )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

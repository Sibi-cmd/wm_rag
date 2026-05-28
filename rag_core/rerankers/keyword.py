"""
rag_core.rerankers.keyword
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reranker that fuses vector similarity with keyword overlap scoring.
"""

from __future__ import annotations

import re
from typing import List

from ..types import SearchResult
from .base import BaseReranker


class KeywordReranker(BaseReranker):
    """Reranker that combines vector similarity score with keyword overlap.

    The final score is a weighted blend:
        ``0.7 * similarity_score + 0.3 * keyword_overlap``

    This catches cases where semantically close but lexically different
    results are ranked too high, and promotes results that share exact
    keywords with the query.
    """

    def __init__(self, similarity_weight: float = 0.7, keyword_weight: float = 0.3) -> None:
        self._sim_weight = similarity_weight
        self._kw_weight = keyword_weight

    # ---- BaseReranker interface ----

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        if not results:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))

        scored: List[tuple[float, SearchResult]] = []
        for result in results:
            text_words = set(re.findall(r"\w+", result.text.lower()))
            overlap = len(query_words & text_words) / max(len(query_words), 1)

            combined_score = (
                self._sim_weight * result.score + self._kw_weight * overlap
            )
            # Create a new SearchResult with the updated score
            reranked = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=combined_score,
                metadata=result.metadata,
            )
            scored.append((combined_score, reranked))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

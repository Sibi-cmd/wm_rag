import re
from typing import List, Dict

class Reranker:
    @staticmethod
    def calculate_rerank_score(query: str, match: dict) -> float:
        """
        Calculates a custom reranking score for retrieved results.
        Combines semantic similarity score, word overlap, section keyword match, and severity boost.
        """
        sim_score = match.get("score", 0.0)
        metadata = match.get("metadata") or {}
        text = metadata.get('text', '').lower()
        query_lower = (query or "").lower()
        
        # Word overlap calculation
        query_words = set(re.findall(r'\w+', query_lower))
        text_words = set(re.findall(r'\w+', text))
        overlap = len(query_words.intersection(text_words)) / max(len(query_words), 1)
        
        # Section alignment boost
        section = metadata.get('section', '').lower()
        section_match = 1.0 if section in query_lower and section else 0.0
        
        # Severity boost
        severity = metadata.get('severity', 'low').lower()
        severity_boost = 1.0 if severity == 'high' else (0.5 if severity == 'medium' else 0.0)
        
        # Weighted calculation
        return (0.6 * sim_score) + (0.2 * overlap) + (0.1 * section_match) + (0.1 * severity_boost)

    def rerank(self, query: str, matches: List[dict], top_n: int = 4) -> List[dict]:
        """
        Filter out system/navigational matches, calculate rerank score, and sort matches.
        """
        if not matches:
            return []

        # Filter out generic navigational/metadata headings
        filtered_matches = [
            m for m in matches 
            if m.get("metadata", {}).get("section", "").upper() not in ["CONTENTS", "INDEX", "PREFACE"]
        ]
        target_matches = filtered_matches if filtered_matches else matches

        scored_matches = []
        for m in target_matches:
            if m.get("metadata"):
                # Use a fresh dict copy to keep logic pure
                m_copy = dict(m)
                # Compute custom score and attach
                m_copy["score"] = self.calculate_rerank_score(query, m_copy)
                scored_matches.append(m_copy)

        scored_matches.sort(key=lambda x: x["score"], reverse=True)
        return scored_matches[:top_n]

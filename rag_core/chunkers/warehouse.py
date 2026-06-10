"""
rag_core.chunkers.warehouse
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Custom chunker incorporating document processing, structural segmentation, and severity/escalation tagging.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from ..config import ChunkerConfig
from ..types import Chunk
from .base import BaseChunker


class WarehouseChunker(BaseChunker):
    """Custom document chunker for warehouse manuals.

    Performs structural headings-based text segmenting, token-aware sub-chunking,
    and automatically infers severity level, escalation flags, and deterministic IDs.

    Args:
        config: A ``ChunkerConfig`` specifying chunk_size and chunk_overlap.
    """

    def __init__(self, config: ChunkerConfig) -> None:
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        try:
            self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-3.5-turbo",
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        except ImportError:
            # Character-based fallback
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size * 4,
                chunk_overlap=self._chunk_overlap * 4,
            )

    @staticmethod
    def infer_severity(text: str) -> str:
        """Infer severity level from keywords."""
        text_lower = text.lower()
        high_keywords = ["danger", "critical", "risk", "failure", "fatal", "severe"]
        medium_keywords = ["warning", "caution", "attention", "important"]

        if any(k in text_lower for k in high_keywords):
            return "high"
        if any(k in text_lower for k in medium_keywords):
            return "medium"
        return "low"

    @staticmethod
    def infer_escalation(text: str, severity: str) -> bool:
        """Infer whether technician escalation is required."""
        if severity == "high":
            return True

        text_lower = text.lower()
        escalation_phrases = [
            "contact service center",
            "authorized technician required",
            "do not attempt repair",
            "see your dealer",
            "professional assistance needed",
        ]
        return any(p in text_lower for p in escalation_phrases)

    # ---- BaseChunker interface ----

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        metadata = metadata or {}
        file_name = metadata.get("source", "unknown")

        # Step 1: Structural Segmentation by Headings
        section_pattern = (
            r"^(?:"
            r"(?:[0-9.]+\s+)?[A-Z][A-Z\s]{3,}"
            r"|(?:[0-9.]+\s+)[A-Z][a-z].*"
            r"|WARNING:|CAUTION:|NOTE:|PROCEDURE:|TROUBLESHOOTING:"
            r")"
        )
        segments = re.split(f"({section_pattern})", text, flags=re.MULTILINE)

        all_chunks: List[Chunk] = []
        current_section = "General"

        # Process text prior to first structural heading
        if segments[0].strip():
            self._process_segment(
                segments[0], current_section, file_name, metadata, all_chunks
            )

        # Process heading + content body pairs
        for i in range(1, len(segments), 2):
            header = segments[i].strip()
            content = segments[i + 1] if i + 1 < len(segments) else ""

            # Update active section heading unless it's a transient warning marker
            if not any(k in header.upper() for k in ["WARNING", "CAUTION", "NOTE", "PROCEDURE"]):
                current_section = header

            combined_text = f"{header}\n{content}"
            self._process_segment(
                combined_text, current_section, file_name, metadata, all_chunks
            )

        return all_chunks

    # ---- Internal segment processing ----

    def _process_segment(
        self,
        text: str,
        section: str,
        file_name: str,
        base_metadata: Dict[str, Any],
        accumulator: List[Chunk],
    ) -> None:
        if not text.strip():
            return

        sub_chunks = self._splitter.split_text(text)

        for chunk_text in sub_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            severity = self.infer_severity(chunk_text)
            escalation = self.infer_escalation(chunk_text, severity)

            # Deterministic issue_id and chunk_id mapping
            file_slug = re.sub(r"[^a-z0-9]", "-", file_name.lower())[:20]
            unique_seed = f"{section}-{len(accumulator)}-{file_name}"
            issue_id = f"ISSUE-{hashlib.md5(unique_seed.encode()).hexdigest()[:8].upper()}"
            chunk_id = f"{file_slug}_{hashlib.md5(chunk_text.encode()).hexdigest()[:10]}"

            chunk_meta = {
                **base_metadata,
                "section": section,
                "severity": severity,
                "escalation_required": escalation,
                "issue_id": issue_id,
                "chunk_index": len(accumulator),
            }

            accumulator.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_meta,
                )
            )

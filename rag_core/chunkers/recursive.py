"""
rag_core.chunkers.recursive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recursive character text splitter with token-aware chunking.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from ..config import ChunkerConfig
from ..types import Chunk
from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    """Token-aware recursive text splitter.

    Splits text by headings / structural markers first, then recursively
    splits oversized segments using the configured chunk size and overlap.

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
            # Fallback to character splitter if tiktoken is not available
            # 1 token ≈ 4 characters
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size * 4,
                chunk_overlap=self._chunk_overlap * 4,
            )

    # ---- BaseChunker interface ----

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        metadata = metadata or {}

        # Step 1: Structural segmentation by headings
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

        # Process leading text (before first heading)
        if segments[0].strip():
            self._process_segment(segments[0], current_section, metadata, all_chunks)

        # Process heading+content pairs
        for i in range(1, len(segments), 2):
            header = segments[i].strip()
            content = segments[i + 1] if i + 1 < len(segments) else ""

            # Update section name unless it's a warning-style marker
            if not any(k in header.upper() for k in ["WARNING", "CAUTION", "NOTE", "PROCEDURE"]):
                current_section = header

            combined = f"{header}\n{content}"
            self._process_segment(combined, current_section, metadata, all_chunks)

        return all_chunks

    # ---- Internal helpers ----

    def _process_segment(
        self,
        text: str,
        section: str,
        metadata: Dict[str, Any],
        accumulator: List[Chunk],
    ) -> None:
        if not text.strip():
            return

        sub_chunks = self._splitter.split_text(text)

        for chunk_text in sub_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = self._make_chunk_id(chunk_text, len(accumulator))
            chunk_meta = {
                **metadata,
                "section": section,
                "chunk_index": len(accumulator),
            }

            accumulator.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_meta,
                )
            )

    @staticmethod
    def _make_chunk_id(text: str, index: int) -> str:
        content_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        return f"chunk_{index}_{content_hash}"

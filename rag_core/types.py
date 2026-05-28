"""
rag_core.types
~~~~~~~~~~~~~~
Core data classes used across all rag_core components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A raw document before chunking.

    Attributes:
        text: The full text content of the document.
        metadata: Arbitrary key-value metadata (source file, page number, etc.).
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document after chunking.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The text content of this chunk.
        metadata: Inherited document metadata plus chunk-specific metadata.
        embedding: The vector embedding (populated after embedding step).
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """A single result returned from a vector store search.

    Attributes:
        chunk_id: The ID of the matched chunk.
        text: The text content of the matched chunk.
        score: Similarity score (higher = more similar).
        metadata: Metadata associated with the chunk.
    """

    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """The final response returned by the RAG pipeline.

    Attributes:
        text: The generated answer from the LLM.
        sources: List of source references (file names, sections, etc.).
        confidence: Confidence score (0.0 to 1.0) based on retrieval similarity.
        chunks_used: Number of context chunks sent to the LLM.
        raw_chunks: The actual SearchResult objects used as context.
    """

    text: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    chunks_used: int = 0
    raw_chunks: List[SearchResult] = field(default_factory=list)

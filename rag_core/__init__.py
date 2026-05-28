"""
rag_core
~~~~~~~~
A standalone, reusable, plug-and-play Retrieval-Augmented Generation (RAG) library.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    ChunkerConfig,
    EmbedderConfig,
    GeneratorConfig,
    RAGConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from .ingest import ingest_directory
from .pipeline import RAGPipeline
from .router import RAGRouter
from .types import Chunk, Document, RAGResponse, SearchResult

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "EmbedderConfig",
    "VectorStoreConfig",
    "GeneratorConfig",
    "ChunkerConfig",
    "RetrievalConfig",
    "Document",
    "Chunk",
    "SearchResult",
    "RAGResponse",
    "ingest_directory",
    "RAGRouter",
]

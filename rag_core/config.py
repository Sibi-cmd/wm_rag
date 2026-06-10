"""
rag_core.config
~~~~~~~~~~~~~~~
Configuration loader that reads a YAML file and resolves environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbedderConfig:
    provider: str = "sentence-transformers"
    model: str = "all-mpnet-base-v2"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"
    index_name: str = "default-index"
    api_key_env: str = "QDRANT_API_KEY"
    # FAISS-specific
    index_path: str = "./faiss_index"
    dimensions: int = 768
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)


@dataclass
class GeneratorConfig:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    max_retries: int = 3
    timeout: int = 30
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)


@dataclass
class ChunkerConfig:
    provider: str = "warehouse"
    chunk_size: int = 500
    chunk_overlap: int = 50
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    rerank: bool = True
    rerank_top_k: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


DEFAULT_PROMPT_TEMPLATE = """\
Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGConfig:
    """Top-level configuration for the entire RAG pipeline."""

    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated RAGConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create a RAGConfig from a plain dictionary (useful for testing)."""
        return cls._from_dict(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "RAGConfig":
        def _pop_known(section_dict: dict, known_keys: set) -> Dict[str, Any]:
            extra = {k: v for k, v in section_dict.items() if k not in known_keys}
            return extra

        cfg = cls()

        # Embedder
        if "embedder" in raw:
            e = raw["embedder"]
            known = {"provider", "model"}
            cfg.embedder = EmbedderConfig(
                provider=e.get("provider", cfg.embedder.provider),
                model=e.get("model", cfg.embedder.model),
                extra=_pop_known(e, known),
            )

        # Vector Store
        if "vector_store" in raw:
            vs = raw["vector_store"]
            known = {"provider", "index_name", "api_key_env", "index_path", "dimensions"}
            cfg.vector_store = VectorStoreConfig(
                provider=vs.get("provider", cfg.vector_store.provider),
                index_name=vs.get("index_name", cfg.vector_store.index_name),
                api_key_env=vs.get("api_key_env", cfg.vector_store.api_key_env),
                index_path=vs.get("index_path", cfg.vector_store.index_path),
                dimensions=vs.get("dimensions", cfg.vector_store.dimensions),
                extra=_pop_known(vs, known),
            )

        # Generator
        if "generator" in raw:
            g = raw["generator"]
            known = {"provider", "model", "api_key_env", "max_retries", "timeout"}
            cfg.generator = GeneratorConfig(
                provider=g.get("provider", cfg.generator.provider),
                model=g.get("model", cfg.generator.model),
                api_key_env=g.get("api_key_env", cfg.generator.api_key_env),
                max_retries=g.get("max_retries", cfg.generator.max_retries),
                timeout=g.get("timeout", cfg.generator.timeout),
                extra=_pop_known(g, known),
            )

        # Chunker
        if "chunker" in raw:
            c = raw["chunker"]
            known = {"provider", "chunk_size", "chunk_overlap"}
            cfg.chunker = ChunkerConfig(
                provider=c.get("provider", cfg.chunker.provider),
                chunk_size=c.get("chunk_size", cfg.chunker.chunk_size),
                chunk_overlap=c.get("chunk_overlap", cfg.chunker.chunk_overlap),
                extra=_pop_known(c, known),
            )

        # Retrieval
        if "retrieval" in raw:
            r = raw["retrieval"]
            known = {"top_k", "rerank", "rerank_top_k"}
            cfg.retrieval = RetrievalConfig(
                top_k=r.get("top_k", cfg.retrieval.top_k),
                rerank=r.get("rerank", cfg.retrieval.rerank),
                rerank_top_k=r.get("rerank_top_k", cfg.retrieval.rerank_top_k),
                extra=_pop_known(r, known),
            )

        # Prompt template
        if "prompt_template" in raw:
            cfg.prompt_template = raw["prompt_template"]

        return cfg

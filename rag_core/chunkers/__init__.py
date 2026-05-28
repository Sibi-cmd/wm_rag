"""
rag_core.chunkers
~~~~~~~~~~~~~~~~~
Chunker registry — maps provider names to concrete classes.
"""

from __future__ import annotations

from ..config import ChunkerConfig
from .base import BaseChunker

_CHUNKER_REGISTRY = {
    "recursive": (
        "rag_core.chunkers.recursive",
        "RecursiveChunker",
    ),
}

# Default chunker when no explicit provider is set in config
_DEFAULT_CHUNKER = "recursive"


def create_chunker(config: ChunkerConfig) -> BaseChunker:
    """Factory that instantiates a chunker from config."""
    provider = config.extra.get("provider", _DEFAULT_CHUNKER)

    if provider not in _CHUNKER_REGISTRY:
        registered = ", ".join(sorted(_CHUNKER_REGISTRY))
        raise ValueError(
            f"Unknown chunker provider '{provider}'. "
            f"Registered providers: {registered}"
        )

    import importlib

    module_path, class_name = _CHUNKER_REGISTRY[provider]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def register_chunker(provider: str, module_path: str, class_name: str) -> None:
    """Register a custom chunker provider at runtime."""
    _CHUNKER_REGISTRY[provider] = (module_path, class_name)

"""
rag_core.rerankers
~~~~~~~~~~~~~~~~~~
Reranker registry — maps provider names to concrete classes.
"""

from __future__ import annotations

from .base import BaseReranker

_RERANKER_REGISTRY = {
    "warehouse": (
        "rag_core.rerankers.warehouse",
        "WarehouseReranker",
    ),
}

_DEFAULT_RERANKER = "warehouse"


def create_reranker(provider: str | None = None) -> BaseReranker:
    """Factory that instantiates a reranker.

    Args:
        provider: Provider name.  Defaults to ``"keyword"``.

    Returns:
        A concrete BaseReranker instance.
    """
    provider = provider or _DEFAULT_RERANKER

    if provider not in _RERANKER_REGISTRY:
        registered = ", ".join(sorted(_RERANKER_REGISTRY))
        raise ValueError(
            f"Unknown reranker provider '{provider}'. "
            f"Registered providers: {registered}"
        )

    import importlib

    module_path, class_name = _RERANKER_REGISTRY[provider]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def register_reranker(provider: str, module_path: str, class_name: str) -> None:
    """Register a custom reranker provider at runtime."""
    _RERANKER_REGISTRY[provider] = (module_path, class_name)

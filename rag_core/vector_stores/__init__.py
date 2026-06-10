"""
rag_core.vector_stores
~~~~~~~~~~~~~~~~~~~~~~
Vector store registry — maps provider names to concrete classes.
"""

from __future__ import annotations

from ..config import VectorStoreConfig
from .base import BaseVectorStore

# Registry: provider name → (module path, class name)
_VECTOR_STORE_REGISTRY = {
    "qdrant": (
        "rag_core.vector_stores.qdrant_store",
        "QdrantStore",
    ),
}


def create_vector_store(config: VectorStoreConfig) -> BaseVectorStore:
    """Factory that instantiates a vector store from config.

    Args:
        config: A VectorStoreConfig with a ``provider`` field.

    Returns:
        A concrete BaseVectorStore instance.

    Raises:
        ValueError: If the provider is not registered.
    """
    provider = config.provider
    if provider not in _VECTOR_STORE_REGISTRY:
        registered = ", ".join(sorted(_VECTOR_STORE_REGISTRY))
        raise ValueError(
            f"Unknown vector store provider '{provider}'. "
            f"Registered providers: {registered}"
        )

    import importlib

    module_path, class_name = _VECTOR_STORE_REGISTRY[provider]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def register_vector_store(provider: str, module_path: str, class_name: str) -> None:
    """Register a custom vector store provider at runtime.

    Args:
        provider: Short name to use in config YAML (e.g., ``"weaviate"``).
        module_path: Fully qualified module path.
        class_name: Class name inside the module.
    """
    _VECTOR_STORE_REGISTRY[provider] = (module_path, class_name)

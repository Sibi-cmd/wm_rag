"""
rag_core.embedders
~~~~~~~~~~~~~~~~~~
Embedder registry — maps provider names to concrete classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import EmbedderConfig
from .base import BaseEmbedder

if TYPE_CHECKING:
    pass

# Registry: provider name → (module path, class name)
_EMBEDDER_REGISTRY = {
    "sentence-transformers": (
        "rag_core.embedders.sentence_transformer",
        "SentenceTransformerEmbedder",
    ),
}


def create_embedder(config: EmbedderConfig) -> BaseEmbedder:
    """Factory that instantiates an embedder from config.

    Args:
        config: An EmbedderConfig with a ``provider`` field.

    Returns:
        A concrete BaseEmbedder instance.

    Raises:
        ValueError: If the provider is not registered.
    """
    provider = config.provider
    if provider not in _EMBEDDER_REGISTRY:
        registered = ", ".join(sorted(_EMBEDDER_REGISTRY))
        raise ValueError(
            f"Unknown embedder provider '{provider}'. "
            f"Registered providers: {registered}"
        )

    module_path, class_name = _EMBEDDER_REGISTRY[provider]

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def register_embedder(provider: str, module_path: str, class_name: str) -> None:
    """Register a custom embedder provider at runtime.

    Args:
        provider: Short name to use in config YAML (e.g., ``"openai"``).
        module_path: Fully qualified module path (e.g., ``"my_app.embedders.openai"``).
        class_name: Class name inside the module (e.g., ``"OpenAIEmbedder"``).
    """
    _EMBEDDER_REGISTRY[provider] = (module_path, class_name)

"""
rag_core.generators
~~~~~~~~~~~~~~~~~~~
Generator registry — maps provider names to concrete classes.
"""

from __future__ import annotations

from ..config import GeneratorConfig
from .base import BaseGenerator

_GENERATOR_REGISTRY = {
    "gemini": (
        "rag_core.generators.gemini",
        "GeminiGenerator",
    ),
}

_DEFAULT_GENERATOR = "gemini"


def create_generator(config: GeneratorConfig) -> BaseGenerator:
    """Factory that instantiates a generator from config."""
    provider = config.provider or _DEFAULT_GENERATOR
    if provider not in _GENERATOR_REGISTRY:
        registered = ", ".join(sorted(_GENERATOR_REGISTRY))
        raise ValueError(
            f"Unknown generator provider '{provider}'. "
            f"Registered providers: {registered}"
        )

    import importlib

    module_path, class_name = _GENERATOR_REGISTRY[provider]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def register_generator(provider: str, module_path: str, class_name: str) -> None:
    """Register a custom generator provider at runtime."""
    _GENERATOR_REGISTRY[provider] = (module_path, class_name)

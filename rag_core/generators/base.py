"""
rag_core.generators.base
~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all LLM generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Interface that every LLM generator must implement.

    A generator takes a fully-formed prompt string and returns
    the LLM's response text.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The complete prompt string to send to the LLM.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.).

        Returns:
            The generated text response.
        """

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async version of ``generate()``.

        The default implementation delegates to the sync method.
        Subclasses should override this for true async support.
        """
        return self.generate(prompt, **kwargs)

"""
rag_core.generators.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~
Generator backed by Google Gemini API (using the google-genai SDK).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, List

from google import genai

from ..config import GeneratorConfig
from .base import BaseGenerator

logger = logging.getLogger("rag_core.generators.gemini")

# Transient error patterns that should trigger retries
_RETRYABLE_PATTERNS = (
    "429", "RESOURCE_EXHAUSTED",
    "503", "UNAVAILABLE", "SERVICE_UNAVAILABLE",
    "500", "INTERNAL",
    "overloaded", "high demand",
)

# Fallback model chain — if the primary model is persistently unavailable
_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient error worth retrying."""
    exc_str = str(exc).upper()
    return any(p.upper() in exc_str for p in _RETRYABLE_PATTERNS)


class GeminiGenerator(BaseGenerator):
    """LLM generator backed by Google Gemini API.

    Args:
        config: A ``GeneratorConfig`` with ``provider="gemini"``.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self._model_name = config.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        self._api_key_env = config.api_key_env
        self._max_retries = config.max_retries or 3
        
        self._client = None
        if self._api_key:
            self._client = genai.Client(api_key=self._api_key)

    def _ensure_client(self) -> None:
        if self._client is None:
            # Recheck environment variables in case they were set dynamically
            self._api_key = self._api_key or os.getenv("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError(
                    f"Gemini API key not found. "
                    f"Please set the environment variable '{self._api_key_env}' or 'GEMINI_API_KEY'."
                )
            self._client = genai.Client(api_key=self._api_key)

    def _get_model_chain(self) -> List[str]:
        """Return the primary model followed by fallbacks."""
        models = [self._model_name]
        for fb in _FALLBACK_MODELS:
            if fb != self._model_name:
                models.append(fb)
        return models

    # ---- BaseGenerator interface ----

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_client()
        last_error = None
        model_chain = self._get_model_chain()

        for model_name in model_chain:
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        **kwargs
                    )
                    if model_name != self._model_name:
                        logger.info(f"Successfully used fallback model: {model_name}")
                    return response.text
                except Exception as exc:
                    last_error = exc
                    if _is_retryable(exc) and attempt < self._max_retries:
                        wait = min(2 ** attempt, 30)
                        logger.warning(
                            f"Retryable error with {model_name} "
                            f"(attempt {attempt}/{self._max_retries}): {exc}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    elif attempt < self._max_retries:
                        wait = min(2 ** (attempt - 1), 10)
                        logger.warning(f"Error (attempt {attempt}/{self._max_retries}): {exc}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        break

            if model_name != model_chain[-1]:
                logger.warning(f"All retries exhausted for {model_name}, trying fallback model...")

        raise RuntimeError(
            f"Gemini API failed after trying all models {model_chain}. Last error: {last_error}"
        )

    async def agenerate(self, prompt: str, **kwargs) -> str:
        self._ensure_client()
        last_error = None
        model_chain = self._get_model_chain()

        for model_name in model_chain:
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        **kwargs
                    )
                    if model_name != self._model_name:
                        logger.info(f"Successfully used fallback model: {model_name}")
                    return response.text
                except Exception as exc:
                    last_error = exc
                    if _is_retryable(exc) and attempt < self._max_retries:
                        wait = min(2 ** attempt, 30)
                        logger.warning(
                            f"Retryable error with {model_name} "
                            f"(attempt {attempt}/{self._max_retries}): {exc}. "
                            f"Retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
                    elif attempt < self._max_retries:
                        wait = min(2 ** (attempt - 1), 10)
                        logger.warning(f"Error (attempt {attempt}/{self._max_retries}): {exc}. Retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        break

            if model_name != model_chain[-1]:
                logger.warning(f"All retries exhausted for {model_name}, trying fallback model...")

        raise RuntimeError(
            f"Gemini API failed after trying all models {model_chain}. Last error: {last_error}"
        )

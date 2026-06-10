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
from typing import Any

from google import genai

from ..config import GeneratorConfig
from .base import BaseGenerator

logger = logging.getLogger("rag_core.generators.gemini")


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

    # ---- BaseGenerator interface ----

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_client()
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    **kwargs
                )
                return response.text
            except Exception as exc:
                last_error = exc
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < self._max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"Rate limited (attempt {attempt}/{self._max_retries}), retrying in {wait}s...")
                    time.sleep(wait)
                elif attempt < self._max_retries:
                    wait = min(2 ** (attempt - 1), 10)
                    logger.warning(f"Error (attempt {attempt}/{self._max_retries}): {exc}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    break

        raise RuntimeError(
            f"Gemini API failed after {self._max_retries} attempts. Last error: {last_error}"
        )

    async def agenerate(self, prompt: str, **kwargs) -> str:
        self._ensure_client()
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    **kwargs
                )
                return response.text
            except Exception as exc:
                last_error = exc
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < self._max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"Rate limited (attempt {attempt}/{self._max_retries}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < self._max_retries:
                    wait = min(2 ** (attempt - 1), 10)
                    logger.warning(f"Error (attempt {attempt}/{self._max_retries}): {exc}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    break

        raise RuntimeError(
            f"Gemini API failed after {self._max_retries} attempts. Last error: {last_error}"
        )

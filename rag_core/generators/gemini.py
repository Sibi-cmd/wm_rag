"""
rag_core.generators.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~
Generator backed by Google Gemini API.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import google.generativeai as genai

from ..config import GeneratorConfig
from .base import BaseGenerator


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
        
        self._model = None
        if self._api_key:
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)

    def _ensure_model(self) -> None:
        if self._model is None:
            # Recheck environment variables in case they were set dynamically
            self._api_key = self._api_key or os.getenv("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError(
                    f"Gemini API key not found. "
                    f"Please set the environment variable '{self._api_key_env}' or 'GEMINI_API_KEY'."
                )
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)

    # ---- BaseGenerator interface ----

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_model()
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._model.generate_content(prompt, **kwargs)
                return response.text
            except Exception as exc:
                last_error = exc
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < self._max_retries:
                    wait = [2, 4][attempt - 1] if attempt - 1 < 2 else 5
                    print(f"[GeminiGenerator] Rate limited (attempt {attempt}), retrying in {wait}s...")
                    time.sleep(wait)
                elif attempt < self._max_retries:
                    print(f"[GeminiGenerator] Error (attempt {attempt}): {exc}")
                    time.sleep(1)
                else:
                    break

        raise RuntimeError(
            f"Gemini API failed after {self._max_retries} attempts. Last error: {last_error}"
        )

    async def agenerate(self, prompt: str, **kwargs) -> str:
        self._ensure_model()
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._model.generate_content_async(prompt, **kwargs)
                return response.text
            except Exception as exc:
                last_error = exc
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < self._max_retries:
                    wait = [2, 4][attempt - 1] if attempt - 1 < 2 else 5
                    print(f"[GeminiGenerator] Rate limited (attempt {attempt}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < self._max_retries:
                    print(f"[GeminiGenerator] Error (attempt {attempt}): {exc}")
                    await asyncio.sleep(1)
                else:
                    break

        raise RuntimeError(
            f"Gemini API failed after {self._max_retries} attempts. Last error: {last_error}"
        )

"""
rag_core.generators.openai
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generator backed by OpenAI and other OpenAI-compatible APIs (Ollama, DeepSeek, Groq, etc.).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

from ..config import GeneratorConfig
from .base import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    """LLM generator backed by OpenAI-compatible APIs.

    Supports custom base URLs to easily integrate with:
        - OpenAI (GPT-4o, GPT-3.5)
        - DeepSeek
        - Ollama (running locally)
        - Groq, OpenRouter, Together AI, etc.

    Args:
        config: A ``GeneratorConfig`` with ``provider="openai"``.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        from openai import AsyncOpenAI, OpenAI

        # Resolve API Key (optional for local models like Ollama)
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self._api_key_env = config.api_key_env
        self._key_missing = False

        # If key is missing, check if it's standard OpenAI cloud endpoint or a local one
        is_openai = "openai.com" in config.extra.get("base_url", "openai.com")
        if not api_key:
            if is_openai:
                self._key_missing = True
                api_key = "mock-openai-key-unset"
            else:
                api_key = "local-no-key-required"

        # Retrieve extra configurations
        base_url = config.extra.get("base_url") or config.extra.get("api_base")
        self._model = config.model
        self._max_retries = config.max_retries
        self._timeout = config.timeout

        # Initialize clients
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    # ---- BaseGenerator interface ----

    def generate(self, prompt: str, **kwargs) -> str:
        if self._key_missing:
            raise ValueError(
                f"OpenAI API key not found. "
                f"Please set the environment variable '{self._api_key_env}' or 'OPENAI_API_KEY'."
            )
        last_error = None
        # Extract direct parameters or use config extras
        temperature = kwargs.pop("temperature", 0.0)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=self._timeout,
                    **kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if "429" in str(exc) and attempt < self._max_retries:
                    wait = 2 ** attempt
                    print(f"[OpenAIGenerator] Rate limited (attempt {attempt}), retrying in {wait}s...")
                    time.sleep(wait)
                elif attempt < self._max_retries:
                    print(f"[OpenAIGenerator] Error (attempt {attempt}): {exc}")
                    time.sleep(1)
                else:
                    break

        raise RuntimeError(
            f"OpenAI-compatible API failed after {self._max_retries} attempts. Last error: {last_error}"
        )

    async def agenerate(self, prompt: str, **kwargs) -> str:
        if self._key_missing:
            raise ValueError(
                f"OpenAI API key not found. "
                f"Please set the environment variable '{self._api_key_env}' or 'OPENAI_API_KEY'."
            )
        last_error = None
        temperature = kwargs.pop("temperature", 0.0)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=self._timeout,
                    **kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if "429" in str(exc) and attempt < self._max_retries:
                    wait = 2 ** attempt
                    print(f"[OpenAIGenerator] Rate limited (attempt {attempt}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < self._max_retries:
                    print(f"[OpenAIGenerator] Error (attempt {attempt}): {exc}")
                    await asyncio.sleep(1)
                else:
                    break

        raise RuntimeError(
            f"OpenAI-compatible API failed after {self._max_retries} attempts. Last error: {last_error}"
        )

"""Shared helpers for configuring Gemini Live connections."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Sequence

from google import genai
from google.genai import types


DEFAULT_MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_SYSTEM_PROMPT = "あなたは優秀な英語AIアシスタントです。"
DEFAULT_MODALITIES: Sequence[str] = ("text",)


class GeminiConfigError(RuntimeError):
    """Raised when the Gemini client cannot be configured."""


def _resolve_api_key(explicit_key: str | None = None) -> str:
    """Return the Gemini API key or raise :class:`GeminiConfigError`."""

    if explicit_key:
        return explicit_key

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    raise GeminiConfigError(
        "Gemini API key is not configured. "
        "Set the GEMINI_API_KEY environment variable or pass api_key explicitly."
    )


@lru_cache
def get_gemini_client(api_key: str | None = None) -> genai.Client:
    """Create (and memoize) a configured Gemini client."""

    resolved_key = _resolve_api_key(api_key)
    return genai.Client(http_options={"api_version": "v1beta"}, api_key=resolved_key)


def build_live_config(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    response_modalities: Iterable[str] | None = None,
) -> types.LiveConnectConfig:
    """Create a :class:`LiveConnectConfig` instance with sensible defaults."""

    instruction = None
    if system_prompt:
        instruction = types.Content(parts=[types.Part(text=system_prompt)])

    modalities = list(response_modalities or DEFAULT_MODALITIES)

    return types.LiveConnectConfig(
        system_instruction=instruction,
        response_modalities=modalities,
    )


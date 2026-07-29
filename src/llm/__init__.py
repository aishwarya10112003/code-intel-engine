"""
LLM factory — returns the right provider based on the LLM_PROVIDER env var.

Default is Groq. To swap providers later, you add an adapter and one `elif` here — the
rest of the app is untouched. (This is the payoff of the LLMClient interface.)
"""
from __future__ import annotations

import os

from .base import LLMClient
from .groq_client import GroqClient


def get_llm() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return GroqClient()
    # Future: elif provider == "gemini": return GeminiClient()
    #         elif provider == "ollama": return OllamaClient()
    #         elif provider == "claude": return ClaudeClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


__all__ = ["LLMClient", "get_llm"]

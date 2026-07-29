"""
The LLMClient interface — a swappable contract for "talk to a language model".

WHY an interface?
  The RAG logic (retrieve chunks → build prompt → generate answer) should NOT care whether
  the model is Groq, Gemini, a local Ollama model, or Claude. By hiding every provider
  behind one small interface, we can swap providers by changing one line of config — the rest
  of the system never changes. The classic "program to an interface, not an implementation."

Two methods:
  * `generate(system, user)` — plain text-in / text-out (used by the RAG pipeline).
  * `chat_with_tools(messages, tools)` — structured tool-calling for the agent (Phase: agent).
    Returns a dict shaped like OpenAI/Groq's response so callers stay provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Every provider adapter implements these two methods."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Send a system instruction + user message, return the model's text reply."""
        raise NotImplementedError

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a multi-turn conversation with tool-definitions, get back the model's next move.

        Returns a normalized dict:
            {
              "content": str | None,           # text the model wrote (may be None if tool-only)
              "tool_calls": [                  # tools the model wants us to run (may be empty)
                {"id": str, "name": str, "arguments": dict},
                ...
              ],
            }
        Implementations translate their provider's raw response into this shape.
        """
        raise NotImplementedError

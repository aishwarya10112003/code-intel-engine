"""
The LLMClient interface — a swappable contract for "talk to a language model".

WHY an interface?
  The RAG logic (retrieve chunks → build prompt → generate answer) should NOT care whether
  the model is Groq, Gemini, a local Ollama model, or Claude. By hiding every provider
  behind one small interface (`generate`), we can swap providers by changing one line of
  config — the rest of the system never changes. This is the classic "program to an
  interface, not an implementation" principle, and it's a great thing to say in an interview.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Every provider adapter implements this one method."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Send a system instruction + user message, return the model's text reply."""
        raise NotImplementedError

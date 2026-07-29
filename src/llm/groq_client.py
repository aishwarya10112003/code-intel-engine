"""
Groq adapter — one concrete implementation of LLMClient.

Groq hosts fast, free open-source models (Llama family). We read the API key from the
GROQ_API_KEY environment variable so no secret is ever hard-coded.
"""
from __future__ import annotations

import os

from groq import Groq

from .base import LLMClient

# A strong, free model on Groq. Override with the GROQ_MODEL env var if you like
# (e.g. "llama-3.1-8b-instant" for faster/cheaper responses).
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                "Get a FREE key at https://console.groq.com (API Keys -> Create), then run:\n"
                '    export GROQ_API_KEY="gsk_...your key..."'
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Low temperature = focused, factual answers (we don't want creativity here;
            # we want the model to stick to the retrieved sources).
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

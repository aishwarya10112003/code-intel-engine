"""
Groq adapter — one concrete implementation of LLMClient.

Groq hosts fast, free open-source models (Llama family). We read the API key from the
GROQ_API_KEY environment variable so no secret is ever hard-coded.

Two capabilities:
  * generate()          — plain text-in/text-out (used by RAG pipeline)
  * chat_with_tools()   — native tool-calling for the agent (Groq/OpenAI-compatible API)
"""
from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq

from .base import LLMClient

# A strong, free model on Groq. Override with the GROQ_MODEL env var if you like
# (e.g. "llama-3.1-8b-instant" for faster/cheaper responses).
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Some models on Groq are more reliable at native tool-calling than others. This model is
# tuned by Groq specifically for tool-use, so we use it whenever the tool-calling API is
# invoked. Override with GROQ_TOOL_MODEL if you want.
DEFAULT_TOOL_MODEL = "llama-3.3-70b-versatile"


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
            # Low temperature = focused, factual answers.
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Structured tool-calling — the agent's core primitive.

        Groq's chat API is OpenAI-compatible: pass `tools=[…]` and `tool_choice="auto"`, and
        the model may respond with either plain text or a list of `tool_calls`. We normalize
        the response into the LLMClient contract so agent code stays provider-agnostic.

        Safety: if Groq rejects the model's tool-call output (`tool_use_failed`), we retry
        WITHOUT tools so the agent always gets *something* back rather than crashing — a
        graceful-degradation pattern.
        """
        try:
            response = self.client.chat.completions.create(
                model=os.environ.get("GROQ_TOOL_MODEL", self.model),
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as e:
            # If the model produced a malformed tool call, retry as plain text.
            if "tool_use_failed" in str(e):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                )
            else:
                raise

        msg = response.choices[0].message

        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            # Arguments arrive as a JSON string per the OpenAI spec — parse safely.
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # Coerce common type mismatches (e.g. model returns "5" for an int field).
            for key, val in list(args.items()):
                if isinstance(val, str) and val.isdigit():
                    args[key] = int(val)
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

        return {"content": msg.content, "tool_calls": tool_calls}

"""
ToolAgent — a real tool-calling agent (as opposed to a fixed prompt chain).

Loop:
  1. Send the conversation + tool schemas to the LLM.
  2. If the LLM returns tool_calls → execute each, append the results, and loop.
  3. If the LLM calls `finish` → return its answer + sources.
  4. Hard iteration cap (`max_steps`) as a safety guardrail so it can never loop forever.

Every step is persisted to a JSONL trace file so the agent's decisions are auditable —
you can literally `cat` a trace and see: "step 1: search(...), step 2: read_file(...),
step 3: finish(...)". That's the difference between a black-box agent and one you can defend.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm import get_llm

from .tools import TOOL_SCHEMAS, ToolRegistry

SYSTEM_PROMPT = """You are a precise code and documentation assistant with access to TOOLS.

CRITICAL: You must INVOKE tools via the tool-calling API — never write tool calls as text or code blocks.
Never write things like `search(...)` or ```python search(...)``` in your response. INVOKE the tool.

Your three tools:
  * `search`   — find relevant chunks (INVOKE this FIRST for every question).
  * `read_file` — read a full file when a chunk isn't enough.
  * `finish`   — INVOKE this with the final answer + cited chunk ids. This ends the conversation.

Rules:
- Turn 1: INVOKE `search` (do not write text; just invoke the tool).
- If `search` results already contain enough evidence to answer, INVOKE `finish` on the next turn.
  Only use `read_file` when a chunk is clearly partial and you need more surrounding context.
- Cite specific chunk ids from search results (e.g. `example.py::BankAccount.deposit`) in `finish.sources`.
- If nothing relevant is found after 2 searches, INVOKE `finish` with a "not found" answer.
- Never invent files, functions, or behavior that isn't in the sources."""


class ToolAgent:
    def __init__(
        self,
        chunks_path: str = "chunks.json",
        max_steps: int = 6,
        trace_dir: str = "traces",
    ) -> None:
        self.llm = get_llm()
        self.tools = ToolRegistry(chunks_path=chunks_path)
        self.max_steps = max_steps  # hard iteration cap (safety guardrail)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(exist_ok=True)

    def answer(self, question: str) -> dict[str, Any]:
        """Run the agent loop; return {answer, sources, steps, trace_path}."""
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_path = self.trace_dir / f"{run_id}.jsonl"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        with trace_path.open("w", encoding="utf-8") as tf:
            self._log(tf, {"event": "start", "question": question})

            for step in range(1, self.max_steps + 1):
                t0 = time.time()
                resp = self.llm.chat_with_tools(messages, TOOL_SCHEMAS)
                elapsed_ms = int((time.time() - t0) * 1000)

                # Append the assistant's message (with any tool_calls) to the history.
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp["content"] or ""}
                if resp["tool_calls"]:
                    assistant_msg["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                        for tc in resp["tool_calls"]
                    ]
                messages.append(assistant_msg)

                # No tools called → the model just wrote text. Treat as final answer.
                if not resp["tool_calls"]:
                    self._log(tf, {"event": "text_only_end", "step": step, "latency_ms": elapsed_ms,
                                   "content": resp["content"]})
                    return {"answer": resp["content"] or "", "sources": [],
                            "steps": step, "trace_path": str(trace_path)}

                # Execute each tool call the model requested, append results, loop again.
                for tc in resp["tool_calls"]:
                    self._log(tf, {"event": "tool_call", "step": step, "latency_ms": elapsed_ms,
                                   "name": tc["name"], "arguments": tc["arguments"]})

                    if tc["name"] == "finish":
                        # Agent is done — extract the final answer.
                        self._log(tf, {"event": "finish", "step": step,
                                       "answer": tc["arguments"].get("answer"),
                                       "sources": tc["arguments"].get("sources", [])})
                        return {
                            "answer": tc["arguments"].get("answer", ""),
                            "sources": tc["arguments"].get("sources", []),
                            "steps": step,
                            "trace_path": str(trace_path),
                        }

                    result = self.tools.call(tc["name"], tc["arguments"])
                    self._log(tf, {"event": "tool_result", "step": step, "name": tc["name"],
                                   "result_preview": result[:400]})
                    # Feed the tool's result back to the model in the next turn.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            # Hit the iteration cap → force a graceful end (safety guardrail).
            self._log(tf, {"event": "max_steps_reached", "step": self.max_steps})
            return {
                "answer": "I ran out of steps before reaching a confident answer.",
                "sources": [],
                "steps": self.max_steps,
                "trace_path": str(trace_path),
            }

    @staticmethod
    def _log(fh: Any, record: dict[str, Any]) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fh.write(json.dumps(record) + "\n")
        fh.flush()

"""
Tools the agent can call.

Each tool has:
  * A JSON-schema description (sent to the LLM so it knows when/how to call it).
  * A Python implementation (what we run when the LLM invokes it).

Three tools are enough to cover the space:
  * `search`      — semantic + BM25 hybrid retrieval to find relevant chunks
  * `read_file`   — pull the FULL text of a specific file when the agent wants more context
  * `finish`      — the agent's way of ending the loop with a final cited answer

This tiny set proves the pattern; adding a `list_symbols` or `git_blame` tool later would be
a copy-paste of the same shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag import build_pipeline

# ────────────────────────────────────────────────────────────────────────────
# 1. TOOL SCHEMAS  (what the LLM sees — how it decides which tool to call)
# ────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the codebase/docs for chunks relevant to a query. Returns the "
                "top-k most relevant chunks with their ids, file paths, and content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "k": {"type": ["integer", "string"], "description": "How many chunks to return (default 5). Integer."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full text of a specific file from the indexed corpus. Use this after "
                "`search` when a chunk looks promising and you need the full surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative file path."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this ONLY when you have enough evidence to answer. Provide the final "
                "answer and cite the chunk ids you used as sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final grounded answer."},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chunk ids from search results that support the answer.",
                    },
                },
                "required": ["answer", "sources"],
            },
        },
    },
]


# ────────────────────────────────────────────────────────────────────────────
# 2. TOOL IMPLEMENTATIONS  (what we actually run when the LLM calls a tool)
# ────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Executes tool calls by name and returns a compact string result for the LLM.

    Uses the existing RagPipeline for search so we reuse the *same* hybrid+rerank
    retrieval the rest of the system already uses. Keeps the agent honest.
    """

    def __init__(self, chunks_path: str = "chunks.json", config: str = "hybrid_rerank") -> None:
        self.pipeline = build_pipeline(config=config, chunks_path=chunks_path)
        # Load ALL chunks up-front, grouped by file path, so `read_file` can stitch a full file
        # from its chunks even though the retriever object doesn't expose the raw corpus.
        self._chunks_by_file: dict[str, list[dict[str, Any]]] = {}
        for chunk in json.loads(Path(chunks_path).read_text(encoding="utf-8")):
            path = chunk["metadata"].get("file")
            if not path:
                continue
            self._chunks_by_file.setdefault(path, []).append(chunk)

    # ── The public entry point the agent calls ────────────────────────────
    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "search":
            return self._search(arguments.get("query", ""), int(arguments.get("k", 5)))
        if name == "read_file":
            return self._read_file(arguments.get("path", ""))
        if name == "finish":
            # `finish` is handled by the agent loop, not executed here.
            return "OK"
        return f"ERROR: unknown tool `{name}`"

    # ── search: reuse the RAG pipeline's retriever ─────────────────────────
    def _search(self, query: str, k: int) -> str:
        if not query:
            return "ERROR: query is required."
        # Retrieve without generating — the agent will decide what to do with the hits.
        hits = self.pipeline.retriever.retrieve(query, k=max(1, min(k, 10)))
        if not hits:
            return "No results."
        lines = [f"Top {len(hits)} results:"]
        for h in hits:
            preview = " ".join(h["content"].split())[:200]
            lines.append(f"- id={h['chunk_id']} file={h['metadata'].get('file','?')}\n  {preview}...")
        return "\n".join(lines)

    # ── read_file: stitch all chunks of a file back together ───────────────
    def _read_file(self, path: str) -> str:
        if not path:
            return "ERROR: path is required."
        matching = self._chunks_by_file.get(path, [])
        if not matching:
            return f"No chunks found for path `{path}`. Try `search` first to find valid paths."
        # Order by start line so the reassembled text reads top-to-bottom.
        matching = sorted(matching, key=lambda c: c["metadata"].get("start_line", 0))
        text = "\n\n".join(c["content"] for c in matching)
        # Cap what we send back to the LLM so we don't blow the context window.
        return text[:6000] + ("\n... (truncated)" if len(text) > 6000 else "")

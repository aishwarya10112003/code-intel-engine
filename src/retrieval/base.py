"""
The Retriever interface — a swappable contract for "find the relevant chunks".

Just like we hid the LLM behind LLMClient, we hide *how* we retrieve behind this interface.
That lets us build several retrievers (dense/vector, keyword/BM25, hybrid) and swap them —
or measure them against each other — without touching the rest of the system.

Every retriever returns a list of "hits". A hit is a plain dict:
    {"chunk_id": str, "content": str, "metadata": dict, "score": float}
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

Hit = dict[str, Any]


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return the top-k most relevant chunks for the query, best first."""
        raise NotImplementedError


def load_corpus(chunks_path: str) -> list[Hit]:
    """Load all chunks from a chunks.json file (used by keyword/BM25 retrieval)."""
    data = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    return [
        {"chunk_id": c["chunk_id"], "content": c["content"], "metadata": c["metadata"]}
        for c in data
    ]

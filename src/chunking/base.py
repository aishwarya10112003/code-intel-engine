"""
The Chunk — the single unit of knowledge in our whole system.

Everything downstream (embeddings, retrieval, the LLM answer) operates on Chunks.
Getting this data shape right *now* saves pain later, so we keep it deliberately small:
an id, the text, and a metadata bag.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    """One logical piece of a file (a function, a class method, a doc section...)."""

    # A stable, human-readable id like "src/app.py::UserService.login".
    # Stable = re-running ingestion on unchanged code produces the same id,
    # which later lets us update/delete chunks precisely instead of rebuilding.
    chunk_id: str

    # The actual text that will be embedded and shown to the LLM.
    content: str

    # Everything else we know about this chunk: file path, language, kind,
    # line numbers, etc. We keep it as a free-form dict so we can add fields
    # (e.g. git blame, imports) later without changing the class.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """For writing to JSON so we can inspect chunks by eye."""
        return asdict(self)

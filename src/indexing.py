"""
Index building as reusable functions (used by the CLI and the deployed web app).

`ensure_index` is the deployment-friendly one: hosted apps have an *ephemeral filesystem*
(the built `.chroma/` may be wiped on restart), so the app calls this on startup to rebuild
the index from the committed `chunks.json` if it's missing. That's why we ship chunks.json in
the repo but not the .chroma folder.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.embeddings import Embedder
from src.store import VectorStore


def build_index(chunks_path: str = "chunks.json") -> int:
    """Embed every chunk in chunks.json and (re)build the vector store. Returns the count."""
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    embedder = Embedder()
    store = VectorStore()
    store.reset()
    store.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embedder.embed_documents([c["content"] for c in chunks]),
        documents=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return store.count()


def ensure_index(chunks_path: str = "chunks.json") -> None:
    """Build the index only if it's empty (e.g. first boot on a fresh host)."""
    if VectorStore().count() == 0:
        build_index(chunks_path)

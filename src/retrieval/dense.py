"""
DenseRetriever — retrieval by *meaning* (the vector search from Phase 2), now wrapped in
the Retriever interface so it can be swapped and combined.

"Dense" refers to the embeddings: they're dense vectors (lots of non-zero numbers), as
opposed to the "sparse" keyword vectors BM25 uses. This is semantic search.
"""
from __future__ import annotations

from src.embeddings import Embedder
from src.store import VectorStore

from .base import Hit, Retriever


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int) -> list[Hit]:
        query_vector = self.embedder.embed_query(query)
        hits = self.store.query(query_vector, n_results=k)
        # Standardize the score key so all retrievers look the same to callers.
        for h in hits:
            h["score"] = h.get("similarity", 0.0)
        return hits

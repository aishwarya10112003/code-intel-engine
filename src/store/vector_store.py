"""
The VectorStore — where we keep the chunk vectors and search them.

WHAT is a vector database?
  Once every chunk is a 384-number vector, we need to store thousands of them AND
  answer "which stored vectors are closest to this query vector?" *fast*. A plain list
  would force us to compare against every vector (slow at scale). A vector database uses
  a smart index (here: HNSW — a graph that finds near neighbours quickly) so search
  stays fast even with lots of chunks. That "find nearest vectors" operation is called
  Approximate Nearest Neighbour (ANN) search.

WHY ChromaDB?
  It's embedded — it runs inside our Python process and saves to a local folder. Zero
  servers to run. Perfect for learning and for a single-machine project. (In production
  you might swap in Qdrant or pgvector; the interface stays the same.)

Distance vs similarity:
  Chroma returns a *cosine distance* (0 = identical meaning, up to 2 = opposite). We
  convert it to a *similarity* score (1 = identical) for readability: similarity = 1 - distance.
"""
from __future__ import annotations

from typing import Any

import chromadb


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only allows str/int/float/bool metadata values (no lists/None).

    Our chunks have a `breadcrumb` list, so we flatten lists to strings and drop Nones.
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, list):
            clean[key] = " > ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    def __init__(self, path: str = ".chroma", collection: str = "code_intel") -> None:
        # PersistentClient saves everything to `path` so the index survives restarts.
        self.client = chromadb.PersistentClient(path=path)
        self._name = collection
        self.collection = self.client.get_or_create_collection(
            name=collection,
            # Tell Chroma to compare vectors with cosine distance (matches our
            # normalized embeddings).
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection — used before a fresh rebuild."""
        try:
            self.client.delete_collection(self._name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store a batch of chunks (their ids, vectors, text, and metadata)."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=[_clean_metadata(m) for m in metadatas],
        )

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict[str, Any]]:
        """Return the `n_results` chunks whose vectors are closest to the query."""
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        # Chroma returns lists-of-lists (one inner list per query); we sent one query.
        for doc, meta, dist, cid in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            res["ids"][0],
        ):
            hits.append(
                {
                    "chunk_id": cid,
                    "content": doc,
                    "metadata": meta,
                    "similarity": round(1.0 - dist, 4),  # distance -> similarity
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

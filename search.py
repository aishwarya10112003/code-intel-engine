"""
search.py — Phase 2, step 2.

Ask a question in plain English; get back the most semantically relevant chunks.
This is "semantic search" — it matches on *meaning*, not exact keywords. Asking
"how do I add money" can find a method called `deposit` even without the word "add".

Run AFTER build_index.py:
    python search.py "how do I deposit money"
    python search.py "where are the delivery rules configured" 8    # top-8
"""
from __future__ import annotations

import sys

from src.embeddings import Embedder
from src.store import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python search.py "your question" [top_k]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    embedder = Embedder()
    store = VectorStore()

    if store.count() == 0:
        print("Index is empty. Run build_index.py first.")
        sys.exit(1)

    query_vector = embedder.embed_query(query)
    hits = store.query(query_vector, n_results=top_k)

    print(f'\nQuery: "{query}"')
    print(f"Top {len(hits)} results:\n" + "-" * 70)
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        location = meta.get("file", "?")
        kind = meta.get("kind", "?")
        # A short preview of the matched text.
        preview = " ".join(hit["content"].split())[:100]
        print(f"{i}. [{hit['similarity']:.3f}]  {hit['chunk_id']}")
        print(f"     kind={kind}  file={location}")
        print(f"     {preview}...")
        print("-" * 70)


if __name__ == "__main__":
    main()

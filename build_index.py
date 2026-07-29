"""
build_index.py — Phase 2, step 1.

Reads the chunks produced by ingest.py (chunks.json), turns each chunk's text into a
vector, and stores everything in the local vector database (ChromaDB).

Run AFTER ingest.py:
    python ingest.py sample_input          # -> chunks.json
    python build_index.py chunks.json      # -> .chroma/  (the searchable index)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.embeddings import Embedder
from src.store import VectorStore


def main() -> None:
    chunks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chunks.json")
    if not chunks_path.exists():
        print(f"Not found: {chunks_path}. Run ingest.py first.")
        sys.exit(1)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        print("No chunks to index.")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    print("Loading embedding model (first run downloads it)...")
    embedder = Embedder()
    store = VectorStore()

    # Fresh rebuild so re-running doesn't create duplicates.
    store.reset()

    # Pull the pieces Chroma needs out of each chunk.
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(documents)} chunks (dimension = {embedder.dimension})...")
    embeddings = embedder.embed_documents(documents)

    print("Storing in the vector database...")
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"\n✓ Index built. {store.count()} chunks are now searchable in ./.chroma/")
    print("Try:  python search.py \"how do I deposit money\"")


if __name__ == "__main__":
    main()

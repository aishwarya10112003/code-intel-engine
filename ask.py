"""
ask.py — Ask your codebase/docs a question and get a cited answer.

Requires a (free) GROQ_API_KEY in your .env. Run AFTER build_index.py:
    python ask.py "how do I deposit money into an account"

Pick the retrieval strategy with the RAG_CONFIG env var (dense | hybrid | hybrid_rerank):
    RAG_CONFIG=hybrid_rerank python ask.py "..."
"""
from __future__ import annotations

import os
import sys

from src.rag import build_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python ask.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    config = os.environ.get("RAG_CONFIG", "dense")
    chunks_path = os.environ.get("CHUNKS_PATH", "chunks.json")

    try:
        pipeline = build_pipeline(config=config, chunks_path=chunks_path)
    except RuntimeError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    answer, hits = pipeline.answer(question)

    print(f"\nQ: {question}    (retrieval: {config})")
    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES (what the answer is grounded in) ===")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit['chunk_id']}   (score {round(hit.get('score', 0.0), 3)})")
    print()


if __name__ == "__main__":
    main()

"""
ask_agent.py — Phase 7. The AGENTIC answer path: decompose → gather → answer → self-check → retry.

Best for complex, multi-part questions. Requires GROQ_API_KEY.
    python ask_agent.py "how do I add money, and what stops me from over-withdrawing?"

Uses the strongest retrieval by default (hybrid_rerank). Override with RAG_CONFIG.
"""
from __future__ import annotations

import os
import sys

from src.agent import AgenticRAG
from src.rag import build_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python ask_agent.py "your (possibly multi-part) question"')
        sys.exit(1)

    question = sys.argv[1]
    config = os.environ.get("RAG_CONFIG", "hybrid_rerank")
    chunks_path = os.environ.get("CHUNKS_PATH", "chunks.json")

    try:
        pipeline = build_pipeline(config=config, chunks_path=chunks_path)
    except RuntimeError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    agent = AgenticRAG(pipeline, max_retries=1)
    answer, hits, trace = agent.answer(question)

    print(f"\nQ: {question}    (retrieval: {config})")
    print(f"\n[agent] split into sub-questions: {trace['sub_questions']}")
    print(f"[agent] self-check faithfulness scores: {trace['faithfulness_scores']}  "
          f"(retries: {trace['retries']})")

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit['chunk_id']}")
    print()


if __name__ == "__main__":
    main()

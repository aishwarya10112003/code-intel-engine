"""
The RAG pipeline — retrieve, augment, generate.

Refactored so it takes a *retriever* (and an optional *reranker*) rather than hard-coding
vector search. This is what lets us plug in dense / hybrid / reranked retrieval and MEASURE
which works best (Phase 4+), without rewriting the answering logic.

Flow:
  1. RETRIEVE  candidate_k chunks with the retriever.
  2. (optional) RERANK them down to top_k with a cross-encoder.
  3. AUGMENT   the prompt with the final chunks as numbered sources.
  4. GENERATE  a cited answer with the LLM.
"""
from __future__ import annotations

from typing import Any

from src.llm import get_llm
from src.retrieval.base import Retriever

SYSTEM_PROMPT = """You are a precise codebase and documentation assistant.

Rules:
- Answer the user's question using ONLY the numbered sources provided. Do not use outside knowledge.
- Cite every claim with its source number in square brackets, e.g. [1], [2].
- If the sources do not contain the answer, reply exactly: "I couldn't find that in the provided sources."
- Be concise and technical. Never invent files, functions, or behavior that isn't in the sources."""


def _format_sources(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        location = meta.get("file", "?")
        name = meta.get("name") or meta.get("heading") or meta.get("kind", "")
        header = f"[{i}] file: {location}" + (f"  ({name})" if name else "")
        blocks.append(f"{header}\n{hit['content']}")
    return "\n\n".join(blocks)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Any = None,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = get_llm()
        self.top_k = top_k
        # We retrieve MORE candidates than we keep, so the (optional) reranker has a good
        # pool to choose the best top_k from. Without a reranker we just take the top_k.
        self.candidate_k = candidate_k if reranker else top_k

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        hits = self.retriever.retrieve(question, k=self.candidate_k)
        if self.reranker is not None:
            hits = self.reranker.rerank(question, hits, top_n=self.top_k)
        else:
            hits = hits[: self.top_k]
        return hits

    def answer(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        hits = self.retrieve(question)
        if not hits:
            return "The index is empty — run build_index.py first.", []
        user_message = f"Sources:\n{_format_sources(hits)}\n\nQuestion: {question}"
        answer = self.llm.generate(SYSTEM_PROMPT, user_message)
        return answer, hits

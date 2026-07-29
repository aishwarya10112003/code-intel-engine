"""
CrossEncoderReranker — a sharp, second-pass filter over retrieved candidates.

WHY rerank at all?
  Retrieval (dense/BM25) is FAST but a bit blurry — it compares the query and each chunk
  *separately* (each was turned into a vector on its own), then measures distance. That scales
  to millions of chunks but sacrifices precision.

  A CROSS-ENCODER is the opposite trade-off: it reads the query and one chunk TOGETHER and
  judges "how well does this chunk answer this query?" — far more accurate, but too slow to
  run over the whole corpus.

THE STANDARD PATTERN (two-stage retrieval):
  1. Retrieval (fast, blurry) narrows millions → ~15 candidates.
  2. Reranker (slow, sharp) reorders those ~15 → the best 5.
  You get accuracy where it matters without paying its cost over everything. This is exactly
  how strong production search systems are built.
"""
from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

# Small, fast, open-source cross-encoder trained for relevance ranking.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """Reorder `hits` by true query-chunk relevance and keep the best `top_n`."""
        if not hits:
            return []
        # The cross-encoder scores (query, chunk) PAIRS — reading both together.
        pairs = [(query, h["content"]) for h in hits]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)[:top_n]

        results: list[dict[str, Any]] = []
        for hit, score in ranked:
            new_hit = dict(hit)
            new_hit["rerank_score"] = float(score)
            new_hit["score"] = float(score)  # keep display consistent
            results.append(new_hit)
        return results

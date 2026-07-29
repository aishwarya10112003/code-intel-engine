"""
HybridRetriever — combine semantic (dense) + keyword (BM25) results.

THE PROBLEM: how do you merge two ranked lists whose scores are on totally different scales?
  Vector similarity is 0-1; BM25 scores are unbounded (could be 0.4 or 40). You can't just
  add them.

THE SOLUTION: Reciprocal Rank Fusion (RRF).
  Ignore the raw scores entirely — use only each item's *rank* (position) in each list.
  Each list contributes  1 / (k + rank)  to an item's fused score. Items that rank highly in
  *either* list bubble to the top; items ranked highly in *both* win overall. Simple,
  scale-free, and surprisingly strong — it's a well-known, respected technique.

  The constant k (default 60) softens the difference between top ranks so rank #1 doesn't
  completely dominate rank #2.
"""
from __future__ import annotations

from .base import Hit, Retriever


class HybridRetriever(Retriever):
    def __init__(self, dense: Retriever, bm25: Retriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[Hit]:
        # Pull a healthy pool from each retriever so fusion has material to work with.
        pool = max(k, 20)
        dense_hits = self.dense.retrieve(query, pool)
        bm25_hits = self.bm25.retrieve(query, pool)

        fused_score: dict[str, float] = {}
        by_id: dict[str, Hit] = {}

        for ranked_list in (dense_hits, bm25_hits):
            for rank, hit in enumerate(ranked_list):
                cid = hit["chunk_id"]
                # +1 because rank is 0-based; RRF wants 1-based positions.
                fused_score[cid] = fused_score.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[cid] = hit

        ordered = sorted(fused_score.items(), key=lambda kv: kv[1], reverse=True)[:k]

        results: list[Hit] = []
        for cid, score in ordered:
            hit = dict(by_id[cid])
            hit["score"] = round(score, 5)  # the RRF score, for display
            results.append(hit)
        return results

"""
BM25Retriever — retrieval by *keywords* (lexical / sparse search).

WHY do we need this if we already have semantic search?
  Semantic (vector) search is great at *meaning* but can miss *exact tokens*. If you search
  for a specific function name like `evaluateServiceability` or an error code `P2002`, you
  want an EXACT match — and that's precisely what keyword search excels at. Code is full of
  exact identifiers, so keyword search complements semantic search perfectly.

WHAT is BM25?
  BM25 is the classic, battle-tested keyword-ranking algorithm (it's what powered search
  engines for decades). It scores a document by how often the query's words appear in it,
  weighted so that rare words count more and long documents don't get an unfair boost.
  "Sparse" because each doc is represented by word counts — mostly zeros.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .base import Hit, Retriever


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word/identifier tokens for keyword matching."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25Retriever(Retriever):
    def __init__(self, corpus: list[Hit]) -> None:
        self.corpus = corpus
        # Pre-tokenize every chunk once and build the BM25 index.
        tokenized = [_tokenize(c["content"]) for c in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(_tokenize(query))
        # Indices of the top-k highest-scoring chunks.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.corpus[i], "score": float(scores[i])} for i in top]

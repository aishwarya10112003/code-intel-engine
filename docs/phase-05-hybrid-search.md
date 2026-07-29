# Phase 5 — Hybrid Search (Keyword + Meaning, Fused)

## 🎯 What this phase does
Adds a second retriever — classic **keyword search (BM25)** — and combines it with the
semantic (vector) search from Phase 2. The combination catches things either one alone would
miss.

## 🧠 The big idea: two kinds of search, each with a blind spot

- **Semantic search (dense/vector)** is great at *meaning*: "add money" → `deposit`. But it
  can fuzz over *exact tokens* — a specific function name, an error code like `P2002`, a
  variable name.
- **Keyword search (BM25/sparse)** is great at *exact words*: search `evaluateServiceability`
  and it finds that exact identifier. But it has no idea that "add money" relates to
  "deposit".

**Code and docs are full of exact identifiers**, so combining both gives the best of each.
This is called **hybrid search**, and it's a hallmark of a serious retrieval system.

---

## 🔩 What we built

```
src/retrieval/bm25.py    → keyword retriever (the BM25 algorithm via rank-bm25)
src/retrieval/hybrid.py  → fuses dense + BM25 results using RRF
```
Plus a new `"hybrid"` config in the pipeline factory, so `evaluate.py hybrid` works.

## 1. BM25 — keyword ranking

**BM25** is the classic, decades-proven keyword ranking algorithm (the engine behind
traditional search). It scores a chunk by how often the query's words appear in it, with two
smart adjustments: **rare words count more** (they're more informative), and **long chunks
don't get an unfair advantage**. It's called "sparse" retrieval because each chunk is
represented by word counts — mostly zeros.

## 2. The hard part: merging two ranked lists (RRF)

Here's the puzzle: dense scores are 0–1 (cosine similarity); BM25 scores are unbounded (could
be 0.4 or 40). **You can't just add them** — they're on different scales.

The elegant fix is **Reciprocal Rank Fusion (RRF)**: throw away the raw scores and use only
each item's **rank** (its position) in each list. Each list contributes `1 / (k + rank)` to
an item's combined score:
```
   fused_score(chunk) = 1/(k + rank_in_dense) + 1/(k + rank_in_bm25)      (k = 60)
```
- A chunk ranked #1 in *either* list gets a strong boost.
- A chunk ranked highly in *both* wins overall.
- Because it uses ranks, not scores, the scale mismatch simply disappears.

RRF is simple, needs no tuning, and is remarkably effective — a great "I chose the robust
standard technique" talking point.

## 3. Result on our sample

On the query *"where is the STRIPE_SECRET_KEY stored?"*, hybrid retrieval put the exact
`API Keys` doc section as source **[1]** — keyword matching nailed the exact identifier. Eval
scores held steady (our tiny corpus was already at ceiling); on a large codebase, hybrid is
where exact-identifier questions stop failing.

---

## 🔑 Words you must know
- **Lexical / keyword / sparse search** — matching on exact words (BM25).
- **Semantic / dense search** — matching on meaning (embeddings).
- **BM25** — the classic keyword ranking algorithm.
- **Hybrid search** — combining keyword + semantic retrieval.
- **RRF (Reciprocal Rank Fusion)** — merging ranked lists using ranks, not raw scores.
- **Sparse vs dense** — word-count vectors (mostly zeros) vs meaning vectors (all non-zero).

---

## 🛡️ Interview defense
> *"Why not just use vector search?"*
> "Vector search matches meaning but can miss exact tokens — function names, error codes,
> identifiers — which are everywhere in code. So I run **hybrid search**: semantic search plus
> **BM25** keyword search. Semantic handles 'add money → deposit'; BM25 handles exact matches
> like a specific function name."

> *"How do you combine two rankings with different score scales?"*
> "**Reciprocal Rank Fusion.** I ignore the raw scores — which are on incomparable scales —
> and fuse by rank: each list contributes 1/(k + rank). A chunk ranked highly in either list
> rises; ranked highly in both, it wins. It's scale-free and needs no tuning, which is why
> it's a standard choice."

**Keywords:** *hybrid retrieval, BM25, lexical vs semantic, sparse vs dense, Reciprocal Rank
Fusion, rank-based fusion, exact-match recall.*

---

## ✅ What you can now say you built
1. A **BM25 keyword retriever** to complement semantic search.
2. A **HybridRetriever** that fuses both with **Reciprocal Rank Fusion**.
3. A new evaluable config (`hybrid`) — measured, no regression.

➡️ Next (Phase 6): **reranking** — a sharper second-pass filter over the hybrid candidates.

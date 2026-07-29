# Phase 6 — Reranking (A Sharper Second-Pass Filter)

## 🎯 What this phase does
Adds a **reranker**: after hybrid search gathers ~15 candidate chunks, a more accurate model
re-reads each candidate *together with the question* and reorders them, so the top few sent
to the LLM are the genuinely best. This is the "two-stage retrieval" pattern used in serious
search systems.

## 🧠 The big idea: fast-but-blurry, then slow-but-sharp

There are two ways a model can compare a question to a chunk:

- **Bi-encoder (what retrieval uses):** turn the question into a vector and each chunk into a
  vector *separately*, then measure distance. Very **fast** and scalable — you can pre-compute
  all the chunk vectors — but a bit **blurry**, because the question and chunk never actually
  "meet".
- **Cross-encoder (the reranker):** feed the question and one chunk into the model
  **together** and get a single relevance score. Much more **accurate**, because it reads
  them jointly — but **slow**, so you can't run it over millions of chunks.

**The trick is to use both in sequence:**
```
   millions of chunks
        │  bi-encoder retrieval (fast)   ← Phases 2 & 5
        ▼
   ~15 candidates
        │  cross-encoder rerank (slow, sharp)   ← this phase
        ▼
   best 5  → sent to the LLM
```
You get cross-encoder accuracy *only* on the small candidate set — accuracy where it counts,
without paying its cost across everything. This is **two-stage retrieval**.

---

## 🔩 What we built

```
src/retrieval/rerank.py  → CrossEncoderReranker (sentence-transformers CrossEncoder)
```
Plus the `"hybrid_rerank"` config: retrieve a **larger** candidate pool (15), then rerank down
to the top 5. (Notice the pipeline retrieves `candidate_k=15` when a reranker is present —
that's on purpose: give the sharp filter a good pool to choose from.)

## How it works in code
For each candidate chunk, we form a `(question, chunk_text)` pair and ask the cross-encoder to
score it. We sort by that score and keep the top few. On *"how do I add money"*, the reranker
scored the `deposit` method **+5.23** and pushed `withdraw` down to **−9.26** — a decisive,
confident ordering. (Cross-encoder scores are raw relevance logits, so they can be negative;
only the *order* matters.)

## When reranking earns its keep
On our tiny sample corpus, retrieval was already good, so reranking mostly confirms the order.
Its real value shows on **large, noisy corpora**, where the first-stage retrieval returns
several plausible-but-wrong chunks and the reranker's joint reading separates the truly
relevant one. That's exactly the situation you'd measure with the Phase 4 harness on a real
repo.

---

## 🔑 Words you must know
- **Reranking** — reordering retrieved candidates with a more accurate model.
- **Bi-encoder** — encodes query and doc separately (fast; used in retrieval).
- **Cross-encoder** — encodes query and doc together (accurate; used in reranking).
- **Two-stage retrieval** — cheap retrieval to narrow, then expensive rerank to sharpen.
- **Candidate pool** — the larger set retrieved so the reranker has options.
- **Relevance score / logit** — the cross-encoder's judgement (order is what matters).

---

## 🛡️ Interview defense
> *"Your retrieval already works — why add a reranker?"*
> "Retrieval uses a **bi-encoder**: it embeds the query and each chunk separately, which is
> fast and scalable but loses precision because they're never compared jointly. A
> **cross-encoder** reads the query and a chunk *together* and scores relevance far more
> accurately — but it's too slow to run over the whole corpus. So I use **two-stage
> retrieval**: fast hybrid retrieval narrows to ~15 candidates, then the cross-encoder reranks
> those to the best 5. Accuracy where it matters, without the cost everywhere."

> *"How would you know it helps?"*
> "My evaluation harness. I compare `hybrid` vs `hybrid_rerank` on the golden set — if context
> recall or faithfulness improves, the reranker earns its place; if not, I drop it. On a large
> repo it typically lifts precision noticeably."

**Keywords:** *reranking, cross-encoder vs bi-encoder, two-stage retrieval, candidate pool,
precision vs latency trade-off.*

---

## ✅ What you can now say you built
1. A **cross-encoder reranker** as a sharp second-pass filter.
2. **Two-stage retrieval** (retrieve wide → rerank narrow) — a production-grade pattern.
3. The `hybrid_rerank` config, ready to be measured against the others.

➡️ Next (Phase 7): the **agentic loop** — the system decomposes hard questions and
self-checks its own answers.

# 05 — Dense Retrieval: Semantic Search Behind the Retriever Interface

*Subsystem: retrieval by *meaning* (vector search), wrapped in a swappable contract. Code:
`src/retrieval/dense.py`, `src/retrieval/base.py`.*

---

## 1. The Claim

> *"Wrapped vector search as a `DenseRetriever` implementing a common `Retriever` interface — embed the
> query, ANN-search the store, and return normalized hits — so it can be combined with keyword search
> and swapped or measured without touching the pipeline."*

---

## 2. First Principles (from zero)

- **Retrieval** = given a question, return the most relevant chunks. It's the "R" in RAG and the single
  biggest lever on answer quality (if the right chunk isn't retrieved, no prompt can save the answer).
- **Dense retrieval** = retrieval by *meaning*, using dense embeddings + vector search. "Dense" because
  the vectors are packed with non-zero numbers (vs sparse keyword vectors, file 06). This is *semantic*
  search: it matches paraphrases, not just shared words.
- **The Retriever interface** = an abstract contract: any retriever has `retrieve(query, k) -> list of
  hits`. A **hit** is a plain dict `{chunk_id, content, metadata, score}`. Standardizing the shape lets
  dense/BM25/hybrid look identical to callers.
- **`k`** = how many results to return (top-k).
- **Score standardization** = every retriever exposes a `score` field so downstream code (fusion,
  rerank, display) treats them uniformly, even though the underlying number differs (cosine similarity
  vs BM25 vs RRF).

---

## 3. How It Actually Works Under the Hood

**Three steps.** `DenseRetriever.retrieve(query, k)` (1) embeds the query with the *query* mode
(`embed_query`, instruction-prefixed, file 03), (2) asks the vector store for the k nearest vectors by
cosine (file 04), and (3) copies each hit's `similarity` into a standard `score` key so it looks like
every other retriever. That's it — thin by design.

**Why it's just a wrapper.** The heavy lifting (embedding, ANN) already lives in `Embedder` and
`VectorStore`. `DenseRetriever` exists to *adapt* those to the `Retriever` interface, so it can be
combined (hybrid, file 07) and swapped/measured (factory + eval). This is the Adapter pattern: give an
existing capability the shape the rest of the system expects.

**Why the interface matters more than the class.** Because the pipeline depends on `Retriever`, not on
`DenseRetriever`, I can pass it a `HybridRetriever` instead and nothing else changes. The interface is
the seam that makes the whole "dense → hybrid → reranked" progression measurable on one golden set.

**Score semantics.** Dense scores are cosine similarities in [0, 1] (higher = closer meaning). They're
comparable *within* dense results but **not** directly comparable to BM25's unbounded scores — which is
the exact problem RRF solves in file 07.

---

## 4. Diagram

### ASCII — dense retrieval, three thin steps
```
  "how do I deposit money?"
        │  1. embed_query (prefix + normalize)   [file 03]
        ▼
  [384-d query vector]
        │  2. store.query(vector, n_results=k)   [file 04: HNSW + cosine]
        ▼
  nearest chunks + cosine similarity
        │  3. hit["score"] = hit["similarity"]   ← standardize the shape
        ▼
  [{chunk_id, content, metadata, score}, ...]   → pipeline / hybrid / rerank
```

### Mermaid — DenseRetriever as an adapter over Embedder + VectorStore
```mermaid
flowchart LR
  Q["query"] --> EQ["Embedder.embed_query"]
  EQ --> QV["query vector"]
  QV --> VS["VectorStore.query(k) — HNSW cosine"]
  VS --> HS["hits with similarity"]
  HS --> STD["score = similarity (standardize)"]
  STD --> OUT["list[Hit] (Retriever interface)"]
  OUT -.same shape as.-> BM25["BM25Retriever (file 06)"]
```

---

## 5. How It Works in Code-Intel Engine (real code)

**The interface every retriever implements (`src/retrieval/base.py`):**
```python
Hit = dict[str, Any]   # {"chunk_id", "content", "metadata", "score"}

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return the top-k most relevant chunks, best first."""
```

**The dense implementation — a thin adapter (`src/retrieval/dense.py`):**
```python
class DenseRetriever(Retriever):
    def __init__(self, embedder, store):
        self.embedder, self.store = embedder, store

    def retrieve(self, query, k):
        query_vector = self.embedder.embed_query(query)      # meaning of the question
        hits = self.store.query(query_vector, n_results=k)   # nearest vectors (file 04)
        for h in hits:
            h["score"] = h.get("similarity", 0.0)            # standardize the score key
        return hits
```

---

## 6. Why I Chose This

- **A common `Retriever` interface** so dense search is one interchangeable strategy among several —
  the foundation for hybrid retrieval and for measuring each approach against the same golden set.
- **Keep `DenseRetriever` thin** (adapter over `Embedder`/`VectorStore`) — single responsibility, easy
  to test, and no duplication of the embedding/ANN logic that already exists.
- **Standardize `score`** so fusion, reranking, and display don't care which retriever produced a hit —
  clean composition.
- **Query-mode embedding** (not document mode) because that's what preserves recall (file 03).

---

## 7. Alternatives + Comparison Table

| Concern | Chosen | Alternative | Why NOT (here) |
|---|---|---|---|
| Abstraction | **Retriever interface + adapter** | Call embed+store directly in the pipeline | Can't add/compare BM25/hybrid without rewriting; interface is the seam that enables it |
| Retrieval type | **Dense (semantic)** as one leg | Dense only, forever | Misses exact identifiers; hybrid pairs it with BM25 (file 07) |
| Hit shape | **Standard dict with `score`** | Each retriever returns its own shape | Fusion/rerank/display would need per-retriever special-casing |
| Query embedding | **`embed_query` (asymmetric)** | Reuse `embed_documents` | Wrong mode silently lowers recall (file 03) |
| k handling | **Caller passes k / candidate_k** | Hard-code a fixed k | The pipeline needs a bigger candidate pool when reranking (file 08) |

---

## 8. Scenarios & Edge Cases

1. **Paraphrased question.** "add funds" retrieves the `deposit` chunk because meaning matches even with
   no shared words — dense's core strength.
2. **Exact identifier query** (`P2002`). Dense may rank it only okay (rare token) → hybrid + BM25 rescue
   it (files 06–07).
3. **k larger than the corpus.** The store simply returns everything it has; no crash.
4. **Empty index.** Returns `[]`; the pipeline surfaces "run build_index.py first" (file 01).
5. **Swap to hybrid.** The pipeline holds a `Retriever`; passing `HybridRetriever` changes retrieval
   with zero pipeline edits — the interface paying off.
6. **Two hits with the same content, different ids.** Both returned; dedup happens later (agent gather,
   file 10) or is irrelevant for a single answer.

---

## 9. How I Verified It

- **`ask.py` prints each hit's `score`** so I can see on-topic chunks ranking above off-topic ones for a
  given question.
- **The eval harness's *context recall*** (file 12) directly measures whether dense retrieval returns
  the answer-bearing chunk — the metric that proves it works, not just feels right.
- **Interchangeability is proven** by `evaluate.py dense` vs `evaluate.py hybrid` running through the
  same pipeline via the factory — only possible because `DenseRetriever` honors the interface (file 01).

---

## 10. Interview Q&A (easy → hard)

**Q (easy). What is dense retrieval?** "Retrieval by meaning: embed the query, find the nearest chunk
vectors by cosine, return them. 'Dense' refers to the packed embedding vectors, as opposed to sparse
keyword vectors."

**Q (easy). Why wrap it in an interface?** "So dense search is one swappable strategy. The pipeline
depends on the `Retriever` contract, so I can pass dense, hybrid, or reranked retrieval without changing
anything, and compare them on the same test set."

**Q (medium). Why standardize the score field?** "Each retriever's native score is different — cosine
similarity, BM25, RRF. Copying it into a common `score` key means fusion, reranking, and the UI treat
all hits uniformly instead of special-casing each retriever."

**Q (medium). Why is `DenseRetriever` so thin?** "Because embedding and ANN already live in `Embedder`
and `VectorStore`. The retriever is an adapter that gives them the `Retriever` shape — single
responsibility, no duplicated logic."

**Q (hard). Dense retrieval's strengths and blind spots?** "Strength: semantic matching — it finds
paraphrases and related concepts with no shared words. Blind spot: exact tokens — a specific function
name or error code can embed weakly, so it may rank poorly. That's exactly why I add BM25 keyword
retrieval and fuse them (file 07)."

**Q (hard). Its scores are 0–1 — can you compare them to BM25's?** "No — BM25 scores are unbounded and
on a different scale, so adding or comparing them directly is meaningless. That scale mismatch is the
reason I fuse with Reciprocal Rank Fusion, which uses only rank, not raw score (file 07)."

**Q (curveball). If retrieval is the biggest quality lever, why not spend it all on dense?** "Because a
single method has blind spots. The highest-ROI move isn't a better single retriever — it's *combining*
complementary ones (dense + BM25) and then reranking. Diversity of signal beats over-tuning one leg."

---

## 11. Traps to Avoid

- ❌ Don't say dense handles exact identifiers well — that's BM25.
- ❌ Don't compare dense scores to BM25 scores directly — different scales (RRF fixes it).
- ❌ Don't bloat the retriever — the embedding/ANN logic belongs in Embedder/VectorStore.
- ❌ Don't forget it uses `embed_query`, not `embed_documents`.
- ❌ Don't describe retrieval as a minor step — it's the dominant lever on answer quality.

---

⬅️ Prev: [`04-vector-store-and-ann.md`](04-vector-store-and-ann.md) ·
➡️ Next: [`06-bm25-keyword-retrieval.md`](06-bm25-keyword-retrieval.md) ·
🔗 Related: [`07-hybrid-retrieval-rrf.md`](07-hybrid-retrieval-rrf.md), [`01-architecture-and-pipeline.md`](01-architecture-and-pipeline.md)

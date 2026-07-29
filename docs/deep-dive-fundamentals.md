# 🔬 Deep-Dive Fundamentals — The Mechanics Beneath the Project

> The other docs explain *what* each part does. This one goes one level deeper — *how the
> underlying machinery actually works* — so you can survive a deep, ML-focused interviewer.
> Each topic ends with **"how deep to go"** so you know when to stop. You don't need to become
> an ML researcher; you need to not freeze when someone asks "but how does that actually work?"

---

## 1. How does an LLM actually work? (transformers, attention)

**The plain version:** An LLM is trained to do one thing — **predict the next token** given
the previous tokens. It saw trillions of tokens during training and learned the statistical
patterns of language and code. To answer, it generates one token at a time, each time asking
"given everything so far, what's the most likely next token?"

**What's inside — the Transformer:** LLMs use an architecture called the **Transformer**. Its
key trick is **attention**: when processing a word, the model looks at all the other words and
decides *which ones matter* for understanding this one. In "the cat sat because **it** was
tired," attention lets "it" attend strongly to "cat". This "which words relate to which" is
computed for every pair of tokens — that's **self-attention**.

**How deep to go:** For a project interview, this is plenty. Say: *"It's a Transformer that
predicts the next token; attention lets each token weigh which other tokens are relevant."* If
pushed: mention that attention computes query/key/value projections and a weighted sum, and
that models stack many attention layers. You do **not** need the matrix math.

**Defense line:** *"An LLM is a Transformer trained for next-token prediction; the attention
mechanism lets it weigh which parts of the input are relevant to each token. I use it as a
reasoning engine over text I retrieve, not as a knowledge store."*

---

## 2. How are embeddings actually *trained*?

**The plain version:** An embedding model is trained with **contrastive learning**. You show
it pairs of texts: **positive pairs** that mean similar things (a question and its correct
answer) and **negative pairs** that don't. The training objective **pulls positive pairs'
vectors closer together and pushes negatives apart**. After millions of such nudges, the model
places similar meanings near each other in vector space — which is exactly the property we
exploit for retrieval.

**Why it generalizes:** Because it saw so many pairs, it learns *why* things are similar
(shared concepts), so it can place *new*, unseen text sensibly too.

**How deep to go:** Naming "contrastive learning with positive and negative pairs" is already a
strong, above-beginner answer. If pushed: mention it optimizes a loss (like InfoNCE/triplet
loss) that minimizes distance for positives and maximizes it for negatives.

**Defense line:** *"Embedding models are trained with contrastive learning — positive pairs are
pulled together and negatives pushed apart — so semantically similar text ends up close in
vector space. That's the property my retrieval relies on."*

---

## 3. Cosine similarity — the actual math, and why cosine

**The math:** Cosine similarity between two vectors A and B is:

```
   cos(θ) =  (A · B) / (|A| × |B|)
             └─ dot product ─┘ / └─ product of their lengths ─┘
```

It measures the **angle** between the vectors, not their length. Result ranges −1 to 1 (for our
non-negative-ish embeddings, effectively 0 to 1). 1 = same direction = same meaning.

**Why cosine, not Euclidean (straight-line) distance?** Because we care about *direction*
(meaning), not *magnitude*. A long document and a short one about the same topic should count as
similar; cosine ignores length, Euclidean would penalize it. **Trick we use:** we *normalize*
all vectors to length 1, which makes cosine similarity equal to just the dot product — faster.

**How deep to go:** The formula + "it measures angle so length doesn't matter" is a complete
answer.

**Defense line:** *"I compare embeddings with cosine similarity — the dot product of normalized
vectors — because it captures directional meaning and ignores text length, which is what you
want for semantic similarity."*

---

## 4. BM25 — how keyword ranking actually scores

**The plain version:** BM25 scores how well a document matches query keywords using three
ingredients:
- **Term Frequency (TF):** the more a query word appears in a document, the higher the score —
  but with *diminishing returns* (the 10th occurrence adds less than the 2nd).
- **Inverse Document Frequency (IDF):** rare words matter more than common ones. Matching
  "P2002" (rare) counts far more than matching "the" (everywhere).
- **Length normalization:** long documents don't get an unfair advantage just for being long.

Combine them and you get a score per document; rank by it.

**How deep to go:** Naming TF, IDF, and length normalization is a strong answer. You don't need
to recite the exact formula, but knowing "TF with saturation × IDF, length-normalized" impresses.

**Defense line:** *"BM25 ranks by term frequency with diminishing returns, weighted by inverse
document frequency so rare terms count more, and normalized for document length. It's ideal for
exact identifiers in code that semantic search can blur."*

---

## 5. HNSW — how the vector index finds neighbours fast

**The problem:** Comparing a query against a million vectors one-by-one is O(n) — too slow.

**The plain version:** **HNSW (Hierarchical Navigable Small World)** builds a **graph** where
each vector is a node connected to its nearest neighbours. It has *layers*: the top layer has a
few nodes with long-range links (like highways), lower layers are denser (like local roads). To
search, you start at the top, greedily hop toward the query through the highways, then drop to
lower layers for fine-grained local search. You reach the neighbourhood in roughly **O(log n)**
hops instead of checking everything.

**Why "approximate"?** It might occasionally miss the true nearest neighbour, trading a tiny bit
of accuracy for a massive speed gain — almost always worth it.

**How deep to go:** "A layered graph you navigate greedily from coarse to fine, ~log n" is an
excellent answer. You don't need construction details.

**Defense line:** *"The vector DB uses HNSW — a multi-layer proximity graph. Search starts at a
sparse top layer and navigates greedily toward the query through denser layers, giving roughly
logarithmic-time approximate nearest-neighbour search instead of a linear scan."*

---

## 6. Bi-encoder vs cross-encoder (retriever vs reranker)

This is *the* deep question about my two-stage retrieval. Know it cold.

**Bi-encoder (used in retrieval):** embeds the query and each document **separately** into
vectors, then compares vectors. Because documents can be embedded *ahead of time* and stored,
querying is just a fast vector lookup. **Fast, scalable, but less precise** — the query and
document never "see" each other during encoding.

**Cross-encoder (used in reranking):** feeds the query and a document **together** into the model
as one input, so the model can directly compare every word of the query against every word of the
document. **Much more accurate, but slow** — you must run the model fresh for every
(query, document) pair, so you can't precompute.

**Why two stages:** Use the fast bi-encoder to narrow millions → ~15 candidates, then the slow
cross-encoder to precisely reorder just those 15. Best of both: scale *and* precision.

**Defense line:** *"Retrieval uses a bi-encoder — query and docs embedded separately, so it's
fast and precomputable but coarse. Reranking uses a cross-encoder — query and doc read together
for high precision but too slow for the whole corpus. So I retrieve fast, then rerank the small
candidate set — two-stage retrieval."*

---

## 7. Chunk size & overlap — the real trade-offs

**Too small:** each chunk lacks context; the answer might be split across chunks and never fully
retrieved. **Too big:** chunks are unfocused, dilute the embedding's meaning, waste context-window
space, and retrieve vaguely. **Overlap** (repeating some text between adjacent chunks) helps an
idea that straddles a boundary survive in at least one chunk.

**My design sidesteps fixed sizes** by chunking on *logical* boundaries (functions, sections), so
each chunk is naturally the "right" size — a complete unit. The fallback chunker uses ~60-line
windows with 10-line overlap for files I can't parse.

**Defense line:** *"Chunk size trades context vs focus — too big dilutes meaning, too small loses
context. I avoid the trade-off by splitting on logical boundaries so each chunk is a complete unit,
with overlapping windows only as a fallback."*

---

## 8. Why 384 dimensions? What does a "dimension" mean?

Each of the 384 numbers is a learned **feature** of meaning — you can loosely imagine one axis
capturing "is this about money?", another "is this code or prose?", etc. (In reality the axes
aren't human-interpretable, but that's the intuition.) **More dimensions** can capture more
nuance but cost more memory and compute; **fewer** are faster but blurrier. 384 is a deliberate
**small-and-fast** choice; big models use 1024+.

**Defense line:** *"Each dimension is a learned feature of meaning. 384 is a small, fast model —
enough nuance for this task; I'd move to a larger-dimension model only if the eval showed
retrieval was the bottleneck."*

---

## 🎯 The meta-point for the interview

You will not be expected to derive transformer math or the BM25 formula in a *project* interview.
What separates a strong candidate is being able to go **one level deeper than the surface** on
each concept and then say honestly *"I can go deeper if useful, but at a high level that's the
mechanism."* That composure — knowing the layer beneath, and knowing where to stop — is what this
doc gives you.

**If you genuinely don't know something an interviewer asks, the best answer is:** *"I'm not
certain of the exact internals, but my understanding is [X] — I'd verify before relying on it."*
Honesty beats bluffing every time.

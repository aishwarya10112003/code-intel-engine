# Phase 2 — Embeddings & Vector Search (Finding Pieces by *Meaning*)

## 🎯 What this phase does
Turns every chunk into a list of numbers that captures its *meaning* (an **embedding**),
stores them in a **vector database**, and lets you search by meaning: ask *"how do I add
money"* and it finds the `deposit` method — even though "deposit" was never in your question.

---

## 🧠 The big idea first: how a computer compares *meaning*

A computer can't compare the meaning of two sentences directly. So we use a trick:

> Convert each piece of text into a list of numbers — a **vector** — using a neural network
> trained so that **texts with similar meaning become vectors that are close together.**

Then "find text with similar meaning" becomes "find nearby vectors" — which is just math
(distance between points). That's the whole foundation of modern search and RAG.

An analogy: imagine every sentence is a dot on a giant map. Sentences about "money" cluster
in one area, sentences about "delivery" in another. Your question is also placed on the map;
the nearest dots are the most relevant chunks.

---

## 🔩 What we built (the files)

```
src/embeddings/embedder.py   → turns text into vectors (the "meaning machine")
src/store/vector_store.py     → stores vectors and finds the nearest ones (ChromaDB)
build_index.py                → chunks.json → embed each → save to the vector DB
search.py                     → your question → nearest chunks (with scores)
```

## 1. The Embedder — the "meaning machine"

**What is an embedding?** A fixed-length list of numbers representing a text's meaning. Our
model outputs **384 numbers** per text. Similar meaning → similar numbers.

**The model:** `BAAI/bge-small-en-v1.5` — a small, fast, open-source embedding model that
runs **locally on your machine** (no API, no cost, no data leaves your computer).

**Two subtle but important details:**
- **Normalization.** We scale every vector to length 1. This makes comparing two vectors a
  single fast operation (a dot product = **cosine similarity**), and keeps scores in a clean
  0–1 range.
- **Queries vs documents (asymmetric embedding).** The bge model works best if you tell it
  whether text is a *question* or a *passage*. Questions get a short instruction prefix;
  passages don't. Using the wrong mode quietly hurts results — so we have `embed_query` and
  `embed_documents`. (This is the kind of detail that makes an interviewer nod.)

## 2. The VectorStore — a vector database (ChromaDB)

**What is a vector database?** A database built to answer one question fast: *"given this
vector, which stored vectors are closest?"*

**Why not a normal list?** With a list you'd compare your query against *every* stored vector
— fine for 11 chunks, terrible for 100,000. A vector DB builds a smart index (an algorithm
called **HNSW** — a navigable graph) that finds near neighbours without checking everything.
This is called **Approximate Nearest Neighbour (ANN)** search — "approximate" because it
trades a tiny bit of accuracy for a huge speed gain.

**Why ChromaDB?** It's **embedded** — it runs inside our Python program and saves to a local
`.chroma/` folder. Zero servers to manage. Perfect for a single-machine project. (In
production you might swap in **Qdrant** or **pgvector**; the concept is identical.)

**Distance vs similarity:** Chroma returns a *distance* (0 = identical meaning). We convert
it to a *similarity* score (1 = identical) for readability: `similarity = 1 − distance`.

**A cleanup detail:** Chroma only allows simple metadata values (string/number/bool), so we
flatten our breadcrumb *list* into a string before storing. Small, real-world glue.

## 3. The two commands (the workflow)

```
python ingest.py sample_input       # Phase 1  → chunks.json
python build_index.py chunks.json   # Phase 2a → embed all chunks, save to .chroma/
python search.py "how do I add money to an account"   # Phase 2b → search
```

**Proof it works** — real output for the query *"how do I add money to an account"*:
```
1. [0.784]  example.py::BankAccount.deposit     ← found it! query said "add", code says "deposit"
2. [0.682]  example.py::BankAccount.withdraw
3. [0.675]  example.py::BankAccount
```
The word "deposit" is **not** in the question. Keyword search would miss it. **Semantic
search finds it by meaning.** That's the magic of embeddings.

---

## 🔑 Words you must know (this phase)

- **Embedding / vector** — a list of numbers representing a text's meaning.
- **Embedding model** — the neural network that produces embeddings (we use bge-small).
- **Dimension** — how many numbers per vector (384 here).
- **Semantic search** — searching by meaning, not exact words.
- **Vector database** — stores vectors and finds nearest ones fast (ChromaDB).
- **Cosine similarity** — the math for "how similar are two vectors" (1 = identical).
- **ANN (Approximate Nearest Neighbour)** — fast "find nearby vectors" search.
- **HNSW** — the specific index algorithm Chroma uses for ANN.
- **Normalization** — scaling vectors to length 1 so comparison is clean and fast.

---

## 🛡️ Interview defense (say these out loud)

> *"How does your search find relevant code without keyword matching?"*
> "I use **semantic search**. Each chunk is converted to a 384-dimensional **embedding**
> with a local sentence-transformers model, stored in a **vector database** (ChromaDB). A
> query is embedded the same way, and I retrieve the chunks whose vectors are closest by
> **cosine similarity**. So 'how do I add money' matches a `deposit` method by *meaning*,
> not words."

> *"Why a vector database instead of just comparing in a loop?"*
> "At scale, comparing a query against every vector is O(n). A vector DB uses an **ANN**
> index — HNSW — to find nearest neighbours in roughly logarithmic time, trading a hair of
> accuracy for a big speed win. ChromaDB is embedded so there's no server to run; the same
> design swaps cleanly to Qdrant or pgvector in production."

> *"Any subtlety in how you embed?"*
> "Two: I **normalize** vectors so similarity is a clean dot product, and I embed **queries
> and documents differently** — the bge model wants an instruction prefix on queries only.
> Getting that asymmetry right measurably improves retrieval."

**Keywords to drop:** *embeddings, semantic/vector search, cosine similarity, ANN, HNSW,
normalized embeddings, asymmetric query/document encoding, embedded vector store.*

---

## ✅ What you can now say you built
1. A local **embedding** pipeline (bge-small, 384-dim, normalized, query/doc-aware).
2. A **vector database** layer over ChromaDB with cosine ANN search.
3. Commands to build the index and run **semantic search**, proven to match by meaning.

➡️ Next (Phase 3): feed the retrieved chunks to Claude and generate an **answer with
citations** — the first complete RAG loop.

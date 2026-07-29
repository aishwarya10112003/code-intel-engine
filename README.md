# 🧠 Code-Intel Engine

**An agentic RAG system that answers architecture questions about any codebase or documentation — with exact source citations.**

Drop in a repository or a folder of docs, ask questions in plain English (*"how does order placement prevent overselling?"*), and get a grounded, cited answer. Built from scratch in Python to demonstrate production-grade Retrieval-Augmented Generation: structural chunking, hybrid retrieval, reranking, an agentic self-correction loop, and a measured evaluation harness.

<!-- 🔗 Live demo: https://your-app.streamlit.app  (add this link tomorrow after deploying — see docs/deployment-guide.md) -->

> 📚 **New to RAG? Start with [`docs/README.md`](docs/README.md)** — a full plain-English learning guide with interview-defense notes for every concept.

---

## ✨ What makes it more than a "chat with PDF" toy

- **AST-aware chunking** — splits code on *logical* boundaries (functions, classes, methods) via Python's Abstract Syntax Tree, and docs by heading structure — not naive fixed-size cuts.
- **Hybrid retrieval** — combines semantic (vector) search with **BM25** keyword search, fused via **Reciprocal Rank Fusion (RRF)**, catching both *meaning* and *exact identifiers*.
- **Two-stage retrieval** — a fast bi-encoder retrieves candidates; a **cross-encoder reranker** sharpens them to the most relevant few.
- **Agentic loop** — decomposes complex multi-part questions into sub-questions, then runs an **LLM self-critic** that verifies the answer is grounded and retries if not (with a hard cap).
- **Grounded, cited answers** — every claim links to a real source; strict prompting + citations fight hallucination.
- **Evaluation harness** — a golden test set scored on **context recall**, **answer correctness**, and **faithfulness** (LLM-as-judge), so every change is *measured*, not guessed.
- **Swappable by design** — the LLM and retriever sit behind interfaces; swap Groq→Claude or ChromaDB→Qdrant in one line.

---

## 🏗️ Architecture

```
   A repo / folder of docs
        │  ingest.py
        ▼
   ┌──────────────┐   AST + heading-aware splitting
   │  CHUNKING    │   (functions, classes, doc sections)
   └──────┬───────┘
          ▼  build_index.py
   ┌──────────────┐   local embedding model (bge-small, 384-dim)
   │  EMBED+STORE │   → ChromaDB vector database
   └──────┬───────┘
          ▼  a question comes in
   ┌────────────────────────────────────────────────────────┐
   │  RETRIEVE                                                │
   │   dense (vectors) ─┐                                     │
   │                    ├─► Reciprocal Rank Fusion ─► rerank  │
   │   BM25 (keywords) ─┘        (RRF)          (cross-encoder)│
   └──────┬─────────────────────────────────────────────────┘
          ▼
   ┌──────────────┐   [agentic mode: decompose → gather → self-critique → retry]
   │  GENERATE    │   LLM (Groq) writes a CITED answer from the retrieved sources
   └──────┬───────┘
          ▼
   Grounded answer + clickable sources
```

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Code parsing | Python `ast` (stdlib) + heading parser |
| Embeddings | `sentence-transformers` (`bge-small-en-v1.5`, local) |
| Vector database | ChromaDB (embedded, HNSW ANN) |
| Keyword search | `rank-bm25` |
| Reranking | cross-encoder (`ms-marco-MiniLM`) |
| LLM | Groq (Llama 3.3) via a swappable `LLMClient` interface |
| Evaluation | custom harness + LLM-as-judge |
| UI | Streamlit |

---

## 📊 Evaluation results (sample corpus)

Measured with the built-in harness on the golden test set (`python evaluate.py <config>`):

| Config | Context Recall | Answer Correctness | Faithfulness |
|---|---|---|---|
| `dense` (vector only) | 1.00 | 1.00 | 0.87 |
| `hybrid` (+ BM25 + RRF) | 1.00 | 1.00 | 0.87 |

*On larger, noisier corpora the harness is how hybrid search and reranking get tuned and regressions are caught.*

---

## 🚀 Getting started

**Prerequisites:** Python 3.11 or 3.12 (not 3.14 — the ML libraries don't support it yet) and a free [Groq API key](https://console.groq.com).

```bash
# 1. Isolated environment + dependencies
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. Add your free Groq key
echo 'GROQ_API_KEY=gsk_...your key...' > .env

# 3. Index a codebase or docs folder
./.venv/bin/python ingest.py sample_input          # → chunks.json  (or: ingest.py /path/to/any/repo)
./.venv/bin/python build_index.py chunks.json      # → .chroma/  (searchable index)

# 4a. Ask from the command line
./.venv/bin/python ask.py "how do I deposit money into an account?"
./.venv/bin/python ask_agent.py "how do I add money, and what stops over-withdrawing?"   # agentic

# 4b. …or launch the web UI
./.venv/bin/streamlit run app.py

# 5. Measure quality
./.venv/bin/python evaluate.py hybrid              # dense | hybrid | hybrid_rerank
```

Point it at anything: `python ingest.py /path/to/any/repo chunks.json`.

---

## 🗂️ Project structure

```
code-intel-engine/
├── ingest.py          # chunk a folder → chunks.json
├── build_index.py     # embed chunks → vector database
├── ask.py             # CLI question → cited answer
├── ask_agent.py       # agentic answer (decompose + self-critique)
├── evaluate.py        # score a config on the golden set
├── app.py             # Streamlit web UI
├── src/
│   ├── chunking/      # AST + heading chunkers
│   ├── embeddings/    # embedding model wrapper
│   ├── store/         # ChromaDB vector store
│   ├── retrieval/     # dense · BM25 · hybrid (RRF) · reranker
│   ├── llm/           # swappable LLM client (Groq)
│   ├── rag/           # RAG pipeline + config factory
│   ├── agent/         # agentic decompose + self-critique loop
│   └── eval/          # LLM-as-judge
├── tests/golden.json  # evaluation test set
└── docs/              # full learning + interview-prep documentation
```

---

## 📚 Documentation

In-depth guide in [`docs/`](docs/README.md): RAG foundations, a per-phase walkthrough, an
[interview Q&A bank](docs/interview-qa.md), [architecture-decision rationale](docs/architecture-decisions.md),
a [deep dive on the ML mechanics](docs/deep-dive-fundamentals.md), and a
[beginner deployment guide](docs/deployment-guide.md).

## 🔭 Roadmap
- `tree-sitter` chunking for multi-language AST support
- Query rewriting for vague questions
- Conversation memory for follow-up questions
- Swap embedded ChromaDB → Qdrant / pgvector for scale

---

*Built as a from-scratch study of production RAG — every component hand-written to understand the mechanics, not to wrap a framework.*

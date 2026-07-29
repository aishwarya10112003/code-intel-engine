# 🧠 Code-Intel Engine — Your Learning Guide

> These docs explain the *Agentic Codebase & Document Intelligence Engine* we're building
> together, in plain English, as if you built it (because it's yours). Read one phase doc
> at a time. Every technical word is defined. Every design choice is explained. Each doc
> ends with **interview-defense** lines you can say out loud.

---

## 🎯 What is this project? (one paragraph)

A tool where you drop in a code repository or a pile of technical documents, and then ask
it questions in plain English — *"how does login work?"*, *"where is the delivery fee
calculated?"* — and it answers you **with exact citations** pointing to the real source.
Under the hood it's a **RAG** system (explained below) with production-grade retrieval and
an agentic self-checking layer.

## 🔑 The single most important word: **RAG**

**RAG = Retrieval-Augmented Generation.** It's the standard way to make an LLM (like
Claude/ChatGPT) answer questions about *your* private data (your code, your docs) that it
was never trained on.

The idea in one line:
> **Don't ask the LLM to remember your data. Find the relevant pieces first, then hand
> them to the LLM and ask it to answer using only those pieces.**

Three steps, always:
1. **Retrieve** — search your data for the chunks most relevant to the question.
2. **Augment** — stuff those chunks into the prompt as context.
3. **Generate** — the LLM writes an answer grounded in that context (and cites it).

This kills two problems: the LLM can't know your private code (retrieval fixes that), and
the LLM tends to make things up ("hallucinate") — grounding it in retrieved text + forcing
citations keeps it honest.

---

## 📖 Read in this order (for a complete beginner)

**Start here → foundations:**
- [**phase-00-rag-foundations.md**](phase-00-rag-foundations.md) — LLMs, tokens, embeddings, what RAG *is*, RAG vs fine-tuning. **Read this first if you're new.**

**Then the 8 build phases** (table below).

**Then the interview-prep references:**
- [**interview-qa.md**](interview-qa.md) — 30+ likely interview questions with model answers. **The most important doc for placement.**
- [**architecture-decisions.md**](architecture-decisions.md) — why each tool, the alternatives, and trade-offs ("why X not Y?").
- [**deep-dive-fundamentals.md**](deep-dive-fundamentals.md) — the mechanics *beneath* the project (transformers, contrastive learning, BM25 math, HNSW, bi- vs cross-encoder). For deep ML interviewers.
- [**glossary.md**](glossary.md) — every term in the project, defined in one place.

---

## 🗺️ The phases (each = working software + a doc)

| Phase | What it adds | Doc |
|---|---|---|
| 0 | **Foundations** — RAG from absolute zero | [phase-00-rag-foundations.md](phase-00-rag-foundations.md) |
| 1 | **Chunking** — split files into meaningful pieces | [phase-01-chunking.md](phase-01-chunking.md) |
| 2 | **Embeddings + vector search** — find pieces by *meaning* | [phase-02-embeddings-search.md](phase-02-embeddings-search.md) |
| 3 | **Generation + citations** — the LLM answers, with sources | [phase-03-generation-citations.md](phase-03-generation-citations.md) |
| 4 | **Evaluation** — measure answer quality with numbers | [phase-04-evaluation.md](phase-04-evaluation.md) |
| 5 | **Hybrid search** — combine keyword + meaning | [phase-05-hybrid-search.md](phase-05-hybrid-search.md) |
| 6 | **Reranking** — a second, sharper filter | [phase-06-reranking.md](phase-06-reranking.md) |
| 7 | **Agentic loop** — the system checks & corrects itself | [phase-07-agentic-loop.md](phase-07-agentic-loop.md) |
| 8 | **UI + wrap-up** — a demo you can show | [phase-08-ui-and-wrapup.md](phase-08-ui-and-wrapup.md) |

## 🧰 The full tech stack (what each tool is FOR)

| Tool | In one sentence |
|---|---|
| **Python** | The language everything is written in. |
| **`ast` (built-in)** | Reads Python code's structure so we can split on functions/classes. |
| **sentence-transformers** | Turns text into meaning-vectors (embeddings), locally & free. |
| **ChromaDB** | A vector database — stores vectors and finds the nearest ones fast. |
| **rank-bm25** | Classic keyword search (for the hybrid phase). |
| **Claude API (anthropic)** | The LLM that writes the final answers and self-checks. |
| **Streamlit** | Builds the simple chat web UI at the end. |

## ▶️ How to run what exists so far

```bash
cd code-intel-engine
python3.12 -m venv .venv                     # one-time: make the environment
./.venv/bin/pip install -r requirements.txt  # one-time: install libraries

./.venv/bin/python ingest.py sample_input        # Phase 1: make chunks.json
./.venv/bin/python build_index.py chunks.json    # Phase 2: build the search index
./.venv/bin/python search.py "how do I add money" # Phase 2: search by meaning
```

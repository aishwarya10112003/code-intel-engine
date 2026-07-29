# 🎯 Code-Intel Engine — Interview Deep-Dive Notes (Defend Every Line)

> **Purpose:** to let you defend *every design decision and line* of this agentic-RAG project in an
> SDE / ML-engineering interview — including **why you did NOT use the alternatives** — without ever
> going blank. Each file takes one subsystem and dissects it to the bone.
>
> **How these differ from the phase docs (`docs/phase-*.md`):** those are the *learning walkthrough*
> (build it phase by phase). **This** folder is the *defense* layer — first principles, under-the-hood
> mechanics, alternative workflows, failure scenarios, verification, and graded interview Q&A. Read a
> phase doc first if a topic is new; use this to become bulletproof.
>
> 🧩 **Companion files:** a project-wide **[MASTER question bank](99-MASTER-interview-questions.md)** and
> a **[TECH-STACK CHEAT SHEET](90-TECH-STACK-CHEATSHEET.md)** (every technology chosen vs. every
> alternative rejected, in comparison tables). Every diagram ships in **both Mermaid + ASCII**.

---

## 🧬 The 11-part anatomy (every subsystem file follows this exactly)

No file is complete unless it has all 11. This is the guarantee against shallow work.

1. **The Claim** — the exact resume phrase this file lets you defend.
2. **First Principles** — what it is, explained from zero (no assumed knowledge).
3. **How It Actually Works Under the Hood** — the real mechanics, not hand-waving.
4. **Diagram** — the flow/architecture as **Mermaid + ASCII**.
5. **How It Works in Code-Intel Engine** — the real code snippet(s) from *this* repo + pseudocode.
6. **Why I Chose This** — the defensible reasoning.
7. **Alternatives + Comparison Table** — what else exists and *why not* (the "why didn't you use X?" answer).
8. **Scenarios & Edge Cases** — concrete walkthroughs, including failure cases.
9. **How I Verified It** — realistic testing / evaluation, so the numbers sound true.
10. **Interview Q&A** — graded easy → hard, with model answers (as many as the topic needs).
11. **Traps to Avoid** — the things that would expose you as bluffing.

### ⚖️ Two standing rules for these notes
- **Dynamic depth.** RAG-heavy topics (chunking, hybrid+RRF, reranking, the agentic loop, evaluation)
  need a *lot* of depth; some need less. Each file goes as deep as *truly defending it* requires.
- **Never skip the basics.** Before the project-specific cleverness, the everyday building blocks are
  explained from zero — vectors, embeddings, cosine similarity, ANN/HNSW, tokens, context windows,
  prompting, hallucination — because interviewers probe those first. That's the **Fundamentals
  primers (Part 0)**.

---

## 📚 The files (each maps to real code in this repo)

### Part 0 — Fundamentals primers (the building blocks used everywhere)
| # | File | Covers (from zero) |
|---|---|---|
| F1 | [`F1-vectors-embeddings-similarity.md`](F1-vectors-embeddings-similarity.md) | Vectors, **embeddings**, cosine similarity vs distance, normalization, dense vs sparse, **ANN & HNSW** |
| F2 | [`F2-llms-tokens-prompting.md`](F2-llms-tokens-prompting.md) | LLMs, **tokens**, context windows, temperature, **prompting**, **hallucination**, why RAG exists, system vs user messages |

### Part A — Ingestion & indexing (turning a repo into a searchable index)
| # | File | Subsystem | Key code it defends |
|---|---|---|---|
| 1 | [`01-architecture-and-pipeline.md`](01-architecture-and-pipeline.md) | End-to-end architecture, the RAG pipeline, interfaces & the config factory | `ingest.py`, `build_index.py`, `src/rag/pipeline.py`, `src/rag/factory.py` |
| 2 | [`02-chunking-ast-and-structural.md`](02-chunking-ast-and-structural.md) | AST (Python) + heading (Markdown) chunking, dispatcher fallback, the `Chunk` shape | `src/chunking/*` |
| 3 | [`03-embeddings-model.md`](03-embeddings-model.md) | Local embedding model, asymmetric query/doc, normalization | `src/embeddings/embedder.py` |
| 4 | [`04-vector-store-and-ann.md`](04-vector-store-and-ann.md) | ChromaDB, HNSW ANN, cosine, metadata rules, distance→similarity | `src/store/vector_store.py`, `src/indexing.py` |

### Part B — Retrieval (finding the right chunks)
| # | File | Subsystem | Key code it defends |
|---|---|---|---|
| 5 | [`05-dense-retrieval.md`](05-dense-retrieval.md) | Semantic (vector) retrieval behind the Retriever interface | `src/retrieval/dense.py`, `base.py` |
| 6 | [`06-bm25-keyword-retrieval.md`](06-bm25-keyword-retrieval.md) | Lexical/sparse BM25 keyword retrieval | `src/retrieval/bm25.py` |
| 7 | [`07-hybrid-retrieval-rrf.md`](07-hybrid-retrieval-rrf.md) | Fusing dense + BM25 with Reciprocal Rank Fusion | `src/retrieval/hybrid.py` |
| 8 | [`08-reranking-cross-encoder.md`](08-reranking-cross-encoder.md) | Two-stage retrieval, cross-encoder reranking | `src/retrieval/rerank.py` |

### Part C — Generation & agency (writing the grounded answer)
| # | File | Subsystem | Key code it defends |
|---|---|---|---|
| 9 | [`09-generation-and-citations.md`](09-generation-and-citations.md) | Grounded prompting, numbered sources, anti-hallucination, citations | `src/rag/pipeline.py`, `ask.py` |
| 10 | [`10-agentic-loop.md`](10-agentic-loop.md) | Decompose → gather → generate → self-critique → retry (capped) | `src/agent/agent.py`, `ask_agent.py` |
| 11 | [`11-llm-abstraction.md`](11-llm-abstraction.md) | Swappable `LLMClient` interface, Groq adapter, provider factory | `src/llm/*` |

### Part D — Quality & delivery
| # | File | Subsystem | Key code it defends |
|---|---|---|---|
| 12 | [`12-evaluation-and-judge.md`](12-evaluation-and-judge.md) | Golden set, context recall / correctness / faithfulness, LLM-as-judge | `evaluate.py`, `src/eval/judge.py`, `tests/golden.json` |
| 13 | [`13-ui-and-deployment.md`](13-ui-and-deployment.md) | Streamlit UI, model caching, ephemeral-FS index rebuild | `app.py`, `src/indexing.py` |

### 🏆 Companions — cheat sheet & whole-project question bank
| File | What it's for |
|---|---|
| [`90-TECH-STACK-CHEATSHEET.md`](90-TECH-STACK-CHEATSHEET.md) | **The tech-stack defense.** For every layer: what I used, what I rejected, and *why*, in clean comparison tables — the fast-recall sheet for "why X and not Y?" |
| [`99-MASTER-interview-questions.md`](99-MASTER-interview-questions.md) | The **combined, cross-cutting** interview set: the 60-second pitch, STAR narrative, RAG system-design, the full "why not X" gauntlet, retrieval/generation/eval combined, failure-mode drills, behavioral, rapid-fire, curveballs, questions to ask *them*, and a warm-up checklist. **Read this last.** |

**Total: 2 fundamentals primers + 13 subsystem deep-dives + cheat sheet + master bank = 17 files (+ this index).**
Suggested order: **F1 → F2 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 90 → 99.**

---

## 🚦 Progress
```
FUNDAMENTALS   [F1] Vectors/embeddings/ANN   [F2] LLMs/tokens/prompting
INGEST+INDEX   [01] Architecture/pipeline    [02] Chunking   [03] Embeddings   [04] Vector store/ANN
RETRIEVAL      [05] Dense   [06] BM25   [07] Hybrid+RRF   [08] Reranking
GENERATION     [09] Generation+citations   [10] Agentic loop   [11] LLM abstraction
QUALITY        [12] Evaluation+judge   [13] UI+deployment
COMPANIONS     [90] Tech-stack cheat sheet   [99] Master question bank
DIAGRAMS       Mermaid + ASCII in every diagram-bearing file
```

---

## 🗺️ The one-sentence spine of the whole project
**"Turn a codebase into logical chunks → embed them into a vector index → for a question, retrieve
with *both* meaning (dense) and keywords (BM25), fuse and rerank the best few → have an LLM write an
answer grounded ONLY in those sources with citations → and, for hard questions, let an agent decompose,
self-check faithfulness, and retry — all of it *measured* against a golden set."**

Memorize that and most answers write themselves.

---

## 🗓️ How to study this folder (a 4-day plan)
- **Day 1 — foundations:** F1, F2, then 01 (architecture) + the spine above.
- **Day 2 — ingest & retrieve:** 02 (chunking — highest yield), 03, 04, then 05/06/07 (dense → BM25 → hybrid).
- **Day 3 — sharpen & generate:** 08 (reranking), 09 (citations/anti-hallucination), 10 (agentic loop).
- **Day 4 — quality & rehearsal:** 11, 12 (evaluation), 13, the **[cheat sheet](90-TECH-STACK-CHEATSHEET.md)**,
  then drill the **[master bank](99-MASTER-interview-questions.md)** out loud until nothing makes you blank.

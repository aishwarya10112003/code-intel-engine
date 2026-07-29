# Phase 8 — The Demo UI & Wrap-Up

## 🎯 What this phase does
Puts a friendly **web chat interface** on top of everything we built, using **Streamlit**, so
you can demo the whole system in a browser instead of the terminal. Then we zoom out and look
at the complete architecture you now own.

## 🧠 Why Streamlit
Streamlit turns a plain Python script into a web app with almost no web code — no HTML, no
JavaScript, no server setup. For AI demos it's the fastest way to get from "works in the
terminal" to "clickable thing I can show an interviewer." (`st.text_input`, `st.write`,
`st.expander` — that's most of it.)

## 🔩 What we built
```
app.py   → the Streamlit UI
```
Features:
- A **question box**; answers render with the sources in expandable panels.
- A **sidebar** to switch retrieval strategy (dense / hybrid / hybrid_rerank) live.
- An **Agentic mode** toggle for hard multi-part questions.
- `@st.cache_resource` so the heavy models load **once**, not on every question.

## ▶️ How to run the demo
```bash
export GROQ_API_KEY="gsk_..."          # (or keep it in .env)
python ingest.py <your repo or docs>   # 1. chunk
python build_index.py chunks.json      # 2. index
./.venv/bin/streamlit run app.py       # 3. open the web app
```

---

## 🗺️ The complete architecture (what you built, end to end)

```
   A folder of code/docs
        │
   [1] CHUNKING        AST + heading splitting → logical chunks           (Phase 1)
        │
   [2] EMBEDDING       each chunk → 384-dim meaning vector                (Phase 2)
        │
   [3] INDEXING        vectors stored in ChromaDB (fast ANN search)       (Phase 2)
        │
   ─────────────── query time ───────────────
        │
   [4] RETRIEVAL       hybrid: vector + BM25 keyword, fused with RRF      (Phase 5)
        │
   [5] RERANKING       cross-encoder sharpens the top candidates          (Phase 6)
        │
   [6] GENERATION      LLM writes a cited answer from the top chunks      (Phase 3)
        │
   [7] AGENTIC LOOP    decompose hard Qs · self-check · retry (capped)    (Phase 7)
        │
   [8] EVALUATION      golden set + metrics prove it all works            (Phase 4)
        │
   [9] UI              Streamlit chat demo                                (Phase 8)
```

Cross-cutting design wins to name in interviews:
- **Swappable interfaces** — `LLMClient` (any model) and `Retriever` (any strategy) mean you
  change one line to swap Groq→Claude or dense→hybrid. Program to interfaces, not implementations.
- **Metric-driven** — every retrieval choice is validated by the evaluation harness.
- **Grounding everywhere** — citations + a faithfulness gate fight hallucination end to end.

---

## 🎤 The 60-second project pitch (memorize this)

> "I built an **Agentic Codebase & Document Intelligence Engine** — a RAG system that answers
> questions about any repo or docs with source citations. It chunks code by **AST** into
> logical units, embeds them into a **vector database**, and retrieves with **hybrid search**
> (semantic + BM25 keyword, fused via **Reciprocal Rank Fusion**) followed by a **cross-encoder
> reranker**. An LLM then generates a **cited, grounded** answer. For complex questions an
> **agentic loop** decomposes the query and **self-checks** the answer against its sources,
> retrying if it's not well-supported. And I measure the whole thing with an **evaluation
> harness** — context recall, answer correctness, and LLM-judged faithfulness — so every
> component is validated, not assumed. The LLM and retriever are behind swappable interfaces,
> so I can change providers or strategies in one line."

That paragraph hits: RAG, AST chunking, embeddings, vector DB, hybrid search, RRF, reranking,
grounding/citations, agentic decomposition + self-critique, and evaluation. It's a complete,
senior-sounding story.

---

## 🔑 Words you must know
- **Streamlit** — turns a Python script into a web app with minimal code.
- **Two-stage retrieval** — retrieve wide, rerank narrow (recap).
- **Grounding** — answers backed by (and cited to) real sources (recap).
- **Program to an interface** — depend on abstractions (LLMClient, Retriever), swap implementations.

## 🛡️ Interview defense
> *"Walk me through your project."* → give the 60-second pitch above, then offer to zoom into
> any box in the architecture diagram. Each phase doc in this folder is your deep-dive for that box.

---

## ✅ You built the whole thing
Chunking → embeddings → vector DB → hybrid retrieval → reranking → cited generation → agentic
self-correction → evaluation → a web UI. That's a genuinely impressive, end-to-end,
production-shaped RAG system. Congratulations — now read the docs until every term above is
yours. 🎓

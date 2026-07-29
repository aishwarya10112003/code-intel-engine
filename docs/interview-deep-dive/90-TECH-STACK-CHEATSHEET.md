# 90 — Tech-Stack Cheat Sheet (What I Used, What I Rejected, and Why)

*The fast-recall defense sheet. Every layer of Code-Intel Engine, the technology I chose, the serious
alternatives, and the one-breath reason I didn't use them. When an interviewer fires "why X and not
Y?", the answer is on this page. Deeper mechanics live in the numbered files `[NN]`.*

> **How to read this:** the **✅ Chosen** column is what's in the repo; **❌ Rejected** lists the real
> alternatives an interviewer might name; **Why not (in one breath)** is your spoken answer. The guiding
> principle throughout: **this is a from-scratch learning build on a single machine with a free budget —
> so I chose the option that (a) teaches the mechanic, (b) has zero/low cost, and (c) hides behind an
> interface so it can be swapped for a production-grade option later.**

---

## 0. The 4 principles behind every choice (say these first)

1. **From-scratch over framework.** I deliberately did **not** use LangChain/LlamaIndex — I hand-wrote
   chunking, fusion, reranking, and the agent loop to *understand* the mechanics, not to wrap a
   black box. `[01]`
2. **Local & free over paid APIs where quality allows.** Embeddings and reranking run locally (no API,
   no cost); only the final generation calls a *free* hosted LLM (Groq). `[03][08][11]`
3. **Program to interfaces.** `LLMClient` and `Retriever` are abstract contracts, so Groq→Claude or
   Chroma→Qdrant is a one-line swap. Every "we could scale this later" answer rests on this. `[11][05]`
4. **Measure, don't guess.** A golden-set evaluation harness scores every config, so each upgrade
   (dense → hybrid → rerank) is *proven*, not asserted. `[12]`

---

## 1. Language & parsing

| Layer | ✅ Chosen | ❌ Rejected alternatives | Why not (in one breath) |
|---|---|---|---|
| Language | **Python 3.12** | Node/TypeScript, Go, Rust | Python is the lingua franca of ML — `sentence-transformers`, `chromadb`, `rank-bm25` are all Python-first; another language means fighting the ecosystem `[01]` |
| Code parsing | **`ast` (stdlib)** | Regex splitting; `tree-sitter`; naive fixed-size cut | Regex can't understand code structure; `ast` is zero-dependency and gives true function/class boundaries. `tree-sitter` is the *right* multi-language upgrade but is heavier to set up — it's on the roadmap `[02]` |
| Doc parsing | **Heading regex + a stack** | Full Markdown AST parser (markdown-it) | I only need section boundaries + breadcrumbs; a one-line heading regex + an ancestor stack does exactly that with no dependency `[02]` |
| Python version | **3.11 / 3.12** | 3.14 (newest) | The ML wheels (torch, sentence-transformers) don't support 3.14 yet — pinning avoids install hell `[13]` |

---

## 2. Embeddings (turning text into vectors)

| Concern | ✅ Chosen | ❌ Rejected | Why not (in one breath) |
|---|---|---|---|
| Embedding model | **`bge-small-en-v1.5` (local, 384-dim)** | OpenAI `text-embedding-3`, Cohere embed | Paid, per-call cost, and sends my code to a third party; bge-small is free, local, fast, and near-SOTA for its size `[03]` |
| Model size | **small (384-dim)** | bge-large/`e5-large` (1024-dim) | Larger = slower + more RAM for a marginal recall gain at this corpus size; small is the right speed/quality point, and swapping up is one line `[03]` |
| Library | **`sentence-transformers`** | Raw HuggingFace `transformers` + manual pooling | sentence-transformers handles pooling/normalization/batching correctly out of the box — reimplementing that is bug-bait `[03]` |
| Query vs doc | **Asymmetric (query gets an instruction prefix)** | Embed query and doc identically | bge is trained asymmetric; using the wrong mode silently drops recall — so I expose `embed_query` vs `embed_documents` `[03]` |
| Similarity | **Cosine on normalized vectors** | Raw dot product / Euclidean on un-normalized | Normalizing makes cosine = dot product (fast) and length-invariant, so long chunks aren't unfairly favored `[F1][03]` |

---

## 3. Vector store & search index

| Concern | ✅ Chosen | ❌ Rejected | Why not (in one breath) |
|---|---|---|---|
| Vector DB | **ChromaDB (embedded)** | Qdrant, Weaviate, Milvus (servers); Pinecone (SaaS) | Those need a running server or a paid account; Chroma runs *in-process* and persists to a folder — zero ops for a single-machine project. The `VectorStore` interface means swapping to Qdrant later is localized `[04]` |
| Index/ANN | **HNSW (Chroma default)** | Flat/brute-force; IVF | Brute-force is O(N) per query (fine for 100s, not 1000s); HNSW gives ~log-time approximate search that stays fast as the corpus grows `[F1][04]` |
| Distance | **Cosine (`hnsw:space`)** | L2/Euclidean | Matches my normalized embeddings; semantic similarity is about direction, not magnitude `[04]` |
| Persistence | **`PersistentClient` → `.chroma/`** | In-memory only | The index must survive restarts; on ephemeral hosts I rebuild from the committed `chunks.json` via `ensure_index` `[04][13]` |

---

## 4. Retrieval strategy

| Concern | ✅ Chosen | ❌ Rejected | Why not (in one breath) |
|---|---|---|---|
| Base retrieval | **Dense (semantic) + BM25 (keyword) → hybrid** | Dense only | Dense misses *exact identifiers* (`P2002`, `evaluateServiceability`); code is full of them, so keyword search is a genuine complement, not redundancy `[05][06][07]` |
| Keyword algo | **BM25 (`rank-bm25`)** | Plain TF-IDF; Elasticsearch | BM25 is TF-IDF's better-calibrated successor (term saturation + length normalization) and needs no server; Elasticsearch is a whole cluster to run `[06]` |
| Fusion | **Reciprocal Rank Fusion (RRF)** | Weighted score sum; learned fusion | Dense scores (0–1) and BM25 scores (unbounded) live on different scales — summing them is meaningless. RRF uses only *rank*, so it's scale-free and needs no tuning/training `[07]` |
| Precision pass | **Cross-encoder reranker (`ms-marco-MiniLM`)** | Trust first-stage ranking; LLM-rerank every candidate | Bi-encoder retrieval is fast but blurry; a cross-encoder reads (query, chunk) together for real relevance. Running it only on ~15 candidates (not the whole corpus) is the standard two-stage trade-off; LLM-reranking everything is slow and costly `[08]` |
| Reranker cost | **Local cross-encoder** | Cohere Rerank API; GPT rerank | Free, local, deterministic; no per-call cost or data egress `[08]` |

---

## 5. LLM & generation

| Concern | ✅ Chosen | ❌ Rejected | Why not (in one breath) |
|---|---|---|---|
| LLM host | **Groq (Llama 3.3 70B)** | OpenAI GPT-4, Anthropic Claude, local Ollama | Groq serves strong open models *free* and *very fast*; GPT-4/Claude cost money per call; Ollama is great but needs a capable local GPU. All hidden behind `LLMClient`, so swapping is one line `[11]` |
| Provider coupling | **`LLMClient` interface + factory** | Call the Groq SDK directly everywhere | Direct calls scatter a vendor across the codebase; one interface means `LLM_PROVIDER=claude` swaps everything by changing one `elif` `[11]` |
| Determinism | **`temperature=0.1`** | Default (~0.7–1.0) | RAG wants faithful, repeatable answers grounded in sources, not creativity; low temperature curbs drift/hallucination `[09]` |
| Grounding | **Strict system prompt + numbered sources + "say I couldn't find it"** | Free-form prompt; fine-tuning | Prompt-level grounding is cheap, transparent, and effective; fine-tuning is expensive overkill for a retrieval-grounded task `[09]` |
| Citations | **`[n]` markers tied to numbered sources** | No citations; footnote post-processing | In-line `[n]` forces the model to attribute each claim and lets the UI show exactly what grounded it — the core anti-hallucination lever `[09]` |

---

## 6. Agent & evaluation

| Concern | ✅ Chosen | ❌ Rejected | Why not (in one breath) |
|---|---|---|---|
| Agent framework | **Hand-written decompose→gather→critique→retry** | LangChain Agents, LangGraph, ReAct/tool-calling | I wanted to *own* and explain the loop; frameworks hide the control flow and add heavy deps. My loop is ~90 lines I can defend fully `[10]` |
| Loop safety | **Hard `max_retries` cap** | Retry until "good enough" | Unbounded self-correction can loop forever and burn tokens; a hard cap guarantees termination `[10]` |
| Self-check | **LLM-as-judge faithfulness (1–5)** | Regex/heuristic checks; human-only review | Faithfulness ("is every claim supported?") is semantic — hard for rules, easy for an LLM; it runs offline so it never slows real users `[10][12]` |
| Eval metrics | **Context recall + answer correctness + faithfulness on a golden set** | "Looks good to me"; BLEU/ROUGE | Vibes don't catch regressions; BLEU/ROUGE measure surface overlap, not grounding. My three metrics target the things that actually matter for RAG `[12]` |
| Test data | **Hand-curated `golden.json` (incl. an unanswerable Q)** | Auto-generated questions only | A small, honest golden set (with a deliberate "should refuse" case) is more trustworthy than many auto-questions of unknown quality `[12]` |

---

## 7. Interface & delivery

| Concern | ✅ Chosen | ❌ Rejected | Why not (in one breath) |
|---|---|---|---|
| UI | **Streamlit** | React/Next.js + a FastAPI backend | Streamlit turns a Python function into a web app in ~80 lines — perfect for a demo; a React+API stack is a whole separate project for no added value here `[13]` |
| Model loading | **`@st.cache_resource` (load once)** | Reload models per request | The embed/rerank models are heavy (~seconds to load); caching loads them once per process so every question is fast `[13]` |
| Secrets | **Env var / `st.secrets` (`GROQ_API_KEY`)** | Hard-coded key | Never commit secrets; env vars work locally (`.env`) and on Streamlit Cloud (secrets box) `[11][13]` |
| Deploy index | **Ship `chunks.json`, rebuild `.chroma/` on boot** | Commit the built `.chroma/` folder | Hosted filesystems are ephemeral and the binary index is large/opaque; `chunks.json` is small, diffable, and `ensure_index` rebuilds on first boot `[04][13]` |

---

## 8. The 12 "why not X?" answers you must know cold

| If they ask… | Say (one line) |
|---|---|
| "Why not LangChain / LlamaIndex?" | "From-scratch to understand the mechanics; frameworks hide the control flow and add heavy deps. Every piece here is code I can defend." `[01]` |
| "Why not just fixed-size chunks?" | "They cut functions in half — you retrieve half a function, useless to the embedder and the LLM. AST/heading chunks are self-contained units." `[02]` |
| "Why not OpenAI embeddings?" | "Cost + data egress; bge-small is free, local, near-SOTA, and swappable if I ever need more." `[03]` |
| "Why not brute-force vector search?" | "O(N) per query doesn't scale; HNSW gives approximate near-neighbours in ~log time." `[04]` |
| "Why hybrid — isn't semantic enough?" | "Semantic misses exact identifiers/error codes; BM25 nails those. Code needs both." `[07]` |
| "Why RRF and not add the scores?" | "The two score scales are incomparable; RRF fuses on *rank*, so it's scale-free and needs no tuning." `[07]` |
| "Why a reranker if retrieval works?" | "Retrieval is fast but blurry; a cross-encoder reads query+chunk together for precision — run only on the top candidates." `[08]` |
| "Why Groq, not GPT-4/Claude?" | "Free, fast, strong open models; and it's behind an interface, so swapping to Claude is one line." `[11]` |
| "Why temperature 0.1?" | "RAG wants faithful, repeatable answers, not creativity; low temperature reduces drift/hallucination." `[09]` |
| "Why build your own agent loop?" | "To own and explain the decompose/critique/retry logic; it's ~90 lines with a hard retry cap." `[10]` |
| "Why LLM-as-judge?" | "Faithfulness is semantic — rules can't measure it, an LLM can; it runs offline so users never pay for it." `[12]` |
| "Why Streamlit, not React?" | "Fastest path from a Python function to a shippable demo; a React+API stack adds no value for this." `[13]` |

---

## 9. Honest limitations (name them before they do)

- **Single-language AST** — only Python gets true AST chunking today; everything else uses the
  line-window fallback. `tree-sitter` is the fix and is on the roadmap. `[02]`
- **Small golden set** — the eval harness is real but the test set is small, so metrics are directional
  on tiny corpora; the value is catching *regressions* as the corpus grows. `[12]`
- **Embedded Chroma doesn't scale horizontally** — it's single-machine; Qdrant/pgvector behind the same
  interface is the scale path. `[04]`
- **No query rewriting / conversation memory yet** — vague or follow-up questions are weaker; both are
  on the roadmap. `[10]`
- **Reranker + agent add latency** — the two-stage + self-critique path is slower; that's why the plain
  pipeline exists for simple questions and the reranker only sees ~15 candidates. `[08][10]`

Naming these is a strength: it shows you know exactly where the system's edges are and how you'd extend it.

---

⬅️ Back to [`00-INDEX.md`](00-INDEX.md) · deeper mechanics in files `[01]`–`[13]` · whole-project drilling
in [`99-MASTER-interview-questions.md`](99-MASTER-interview-questions.md)

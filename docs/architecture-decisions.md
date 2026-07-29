# 🏛️ Architecture Decisions & Trade-offs

> For every significant choice: what we picked, the alternatives, *why*, and when you'd choose
> differently. Interviewers love "why did you choose X over Y?" — this is your cheat sheet.
> A good engineer doesn't just know *what* they built; they know *what they didn't* and why.

---

## 1. Chunking: AST + headings (not fixed-size)

- **We chose:** AST parsing for Python (functions/classes/methods) + heading-based splitting for docs, with a line-window fallback.
- **Alternatives:** fixed-size character/token chunks; recursive character splitting (LangChain default); tree-sitter for all languages.
- **Why:** logical chunks are complete and self-contained → far better retrieval. Fixed-size cuts functions in half.
- **When you'd differ:** for many languages at once, **tree-sitter** gives real AST parsing everywhere (my planned upgrade). For plain prose with no structure, recursive splitting is fine.
- **Trade-off:** AST is language-specific work; the fallback covers the rest so nothing is dropped.

## 2. Embedding model: `bge-small-en-v1.5` (local, open-source)

- **We chose:** a small (384-dim) open-source model that runs locally.
- **Alternatives:** OpenAI `text-embedding-3` (API, paid, higher quality); larger bge/e5 models (slower, better).
- **Why:** free, private (nothing leaves the machine), fast, and good enough for the task.
- **When you'd differ:** if retrieval quality is the bottleneck on a hard corpus, a larger model or a paid API embedding improves recall — measure it with the eval harness first.
- **Trade-off:** small model = speed + free vs. a few points of accuracy.

## 3. Vector database: ChromaDB (embedded)

- **We chose:** ChromaDB — runs *inside* the Python process, saves to a local folder.
- **Alternatives:** **Qdrant** / **Weaviate** (standalone servers, production-scale), **pgvector** (Postgres extension), **Pinecone** (managed cloud, paid), **FAISS** (library, no persistence layer).
- **Why:** zero infrastructure — no server to run — ideal for a single-machine project and learning.
- **When you'd differ:** at real scale or multi-user, a server like **Qdrant** or **pgvector** (nice if you already run Postgres). The `VectorStore` interface means swapping is localized.
- **Trade-off:** embedded simplicity vs. production features (sharding, replication, filtering at scale).

## 4. Retrieval: hybrid (dense + BM25) with RRF

- **We chose:** semantic + keyword, fused by Reciprocal Rank Fusion.
- **Alternatives:** dense only (simpler, misses exact tokens); a weighted score blend (needs score normalization + tuning).
- **Why:** code has exact identifiers that keyword search catches and semantics miss; RRF fuses without needing to normalize incomparable scores.
- **When you'd differ:** for pure prose with no exact-token needs, dense alone may suffice.
- **Trade-off:** a bit more complexity for meaningfully better recall on code.

## 5. Reranking: cross-encoder, two-stage

- **We chose:** retrieve ~15 candidates fast, rerank to top 5 with a cross-encoder.
- **Alternatives:** no reranker (faster, less precise); a paid rerank API (Cohere Rerank).
- **Why:** cross-encoders are much more accurate but too slow for the whole corpus — two-stage gives accuracy where it counts.
- **When you'd differ:** if latency is critical and retrieval is already precise, skip it — and prove the decision with the eval harness.
- **Trade-off:** extra latency on the candidate set vs. higher precision.

## 6. LLM: Groq (free), behind a swappable interface

- **We chose:** Groq's free API (Llama models), abstracted behind `LLMClient`.
- **Alternatives:** Claude / OpenAI (paid, higher quality), local Ollama (free, offline, weaker small models), Gemini (free tier).
- **Why:** free, fast, no credit card; the interface means the provider is a one-line swap.
- **When you'd differ:** for the best answer quality, a top model like Claude; for fully offline, Ollama.
- **Trade-off:** free/fast vs. the extra quality of a frontier model.

## 7. Orchestration: hand-written (not LangGraph/LangChain)

- **We chose:** plain Python — a small pipeline and a small agent loop I wrote.
- **Alternatives:** LangChain / LlamaIndex / LangGraph (frameworks that provide these pieces).
- **Why:** I understand every step, which matters for interviews and debugging; frameworks hide the mechanics behind abstractions.
- **When you'd differ:** a large team/product benefits from a framework's ecosystem and integrations.
- **Trade-off:** a little more code I own vs. losing visibility into how it works.
- **Great interview line:** *"I built the agent loop and retrieval myself instead of using LangGraph, specifically so I understand and can reason about every step."*

## 8. Evaluation: custom harness + LLM-as-judge (not just vibes)

- **We chose:** a golden set with deterministic metrics + an LLM faithfulness judge.
- **Alternatives:** RAGAS (a ready-made RAG eval library); manual eyeballing (what most people do).
- **Why:** building it myself teaches what the metrics mean and keeps it lightweight; it's the backbone of metric-driven development.
- **When you'd differ:** RAGAS is great for a richer metric suite once the basics are in place.
- **Trade-off:** a small custom harness I fully understand vs. a heavier library.

---

## The meta-principle behind all of it

Two ideas recur:
1. **Program to interfaces, not implementations.** `LLMClient` and `Retriever` mean every big
   dependency (model, vector DB, retrieval strategy) is swappable in one place. That's what
   lets me start free and simple, then upgrade any piece without a rewrite.
2. **Measure, don't guess.** Every choice above can be *validated* with the eval harness. When
   an interviewer pushes "why X?", the strongest answer ends with *"...and I can prove it with
   my evaluation set."*

**Say this if asked "what would you do differently at scale?":**
> "Swap embedded ChromaDB for a server like Qdrant, move to tree-sitter chunking for
> multi-language, batch the embedding pipeline, cache aggressively, and reserve the cross-encoder
> and agent for where the eval shows they pay off. Because everything's behind interfaces, each
> of those is a localized change, not a rewrite."

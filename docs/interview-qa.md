# 🎤 Interview Q&A — Every Question You'll Likely Get

> Model answers to the questions interviewers actually ask about a RAG project. Read them,
> then practice saying them in your own words. Grouped by theme.

---

## A. Foundations (they test if you *understand*, not just built)

**Q: What is RAG, in one sentence?**
"Retrieval-Augmented Generation — instead of relying on the LLM's memory, I retrieve the most
relevant chunks of my data at query time, put them in the prompt, and have the LLM answer from
them with citations."

**Q: Why not just fine-tune the model on the codebase?**
"Fine-tuning bakes knowledge into the model's weights — expensive, needs retraining to update,
and gives no citations. RAG injects knowledge at query time, so new data is searchable
instantly and every answer is traceable to a source. For codebase Q&A — changing data that
needs citations — RAG is the right tool. Fine-tuning is for teaching *style or behavior*, not
*facts*."

**Q: What's an embedding?**
"A vector of numbers that represents a text's meaning, produced by an embedding model trained
so similar meanings map to nearby vectors. It turns 'find relevant text' into 'find nearby
vectors' — pure distance math."

**Q: How does semantic search find `deposit` when I search 'add money'?**
"Both phrases are embedded into vectors by the same model. Because the model learned they mean
similar things, their vectors are close, so 'add money' retrieves the `deposit` chunk even
though the words differ. Keyword search would miss it."

**Q: What's hallucination and how do you prevent it?**
"When the model confidently states something false, because it predicts plausible text rather
than looking up truth. I prevent it with grounding — a strict prompt to answer only from
provided sources — plus mandatory citations, low temperature, and an automated faithfulness
check in my evaluation and agent."

---

## B. Architecture & design choices

**Q: Walk me through your system end to end.**
"A folder gets **chunked** — code by its AST into functions/classes, docs by headings. Each
chunk is **embedded** into a 384-dim vector and stored in **ChromaDB**. At query time I run
**hybrid retrieval** — semantic search plus BM25 keyword search, fused with Reciprocal Rank
Fusion — then a **cross-encoder reranker** sharpens the top candidates. The best chunks go to
the **LLM**, which writes a **cited** answer. For complex questions an **agentic loop**
decomposes the query and self-checks the answer. I validate the whole thing with an
**evaluation harness**."

**Q: Why chunk by AST instead of fixed size?**
"Fixed-size splitting cuts functions in half, so retrieval returns fragments. Parsing the AST
splits on logical boundaries — whole functions, classes, methods — so every chunk is complete
and self-contained. Chunk quality is one of the biggest levers on RAG quality."

**Q: Why hybrid search — isn't vector search enough?**
"Vector search matches meaning but can miss exact tokens like a specific function name or error
code, which are everywhere in code. BM25 keyword search nails exact matches. Combining them
covers both, and I fuse the two ranked lists with Reciprocal Rank Fusion so their different
score scales don't matter."

**Q: What is Reciprocal Rank Fusion and why use it?**
"A way to merge two ranked lists using only each item's rank, not its raw score — each list
contributes 1/(k+rank). It's scale-free, so I don't have to normalize incomparable scores
(cosine 0–1 vs unbounded BM25), and it's a proven, tuning-free standard."

**Q: What's the difference between your retriever and your reranker?**
"The retriever uses a **bi-encoder** — it embeds query and chunk separately, which is fast and
scalable but blurry. The reranker uses a **cross-encoder** — it reads the query and a chunk
*together* for a far more accurate relevance score, but it's slow. So I do two-stage retrieval:
fast retrieval narrows to ~15 candidates, then the cross-encoder reranks to the best 5."

**Q: What makes it 'agentic'?**
"The program makes its own decisions instead of following one fixed path. For a complex
question it decomposes it into sub-questions, retrieves for each, then runs a self-critique —
an LLM judges whether the answer is supported by the sources — and retries with more context if
not, up to a hard cap."

---

## C. Evaluation (your strongest differentiator — lean into it)

**Q: How do you know your system is any good?**
"I measure it. I built an evaluation harness with a golden test set and three metrics: context
recall (did we retrieve the right chunk), answer correctness (does the answer contain the key
fact), and faithfulness (is the answer supported by its sources, graded by an LLM-as-judge).
Every retrieval change is validated against these numbers."

**Q: What is LLM-as-judge? Isn't it circular to grade an LLM with an LLM?**
"Judging is easier than generating — checking 'is this answer supported by this text?' is a
simpler, more reliable task than producing the answer. It runs offline during evaluation only,
and I pair it with deterministic checks (recall, keyword correctness) so I'm not relying on the
judge alone."

**Q: How would you prove reranking actually helped?**
"Run `evaluate.py hybrid` and `evaluate.py hybrid_rerank` on the golden set and compare context
recall and faithfulness. If they improve, reranking earns its place; if not, I drop it. That's
metric-driven development — no guessing."

---

## D. Scale, cost, and production (senior-level thinking)

**Q: How would this scale to 10 million documents?**
"Three things. **Retrieval** already scales — the vector DB uses an ANN index (HNSW), which is
roughly logarithmic, and I'd move from embedded ChromaDB to a server like Qdrant or pgvector.
**Ingestion** I'd batch and parallelize embedding. **Cost/latency** I'd cache embeddings, and
keep the expensive cross-encoder only on the small candidate set. The agent's extra LLM calls
I'd reserve for genuinely hard questions."

**Q: What's the latency of a query?**
"Embedding the query is milliseconds, vector search is milliseconds, reranking ~15 chunks is
fast, and the LLM call dominates — usually the bulk of the time. The agentic path is slower
because it makes several LLM calls, so I use it only for complex questions."

**Q: How do you keep the index fresh when code changes?**
"Chunks have stable IDs, so I can re-index only changed files and upsert those chunks instead of
rebuilding everything. In production I'd trigger that on a git commit hook or a schedule."

**Q: What does it cost to run?**
"Embeddings and reranking are free — they run locally. The only paid part is the LLM, and I use
Groq's free tier; a query is a few cents at most on a paid provider. Caching answers to repeated
questions would cut it further."

---

## E. Weaknesses & follow-ups (have answers ready — they WILL ask)

**Q: What are the limitations of your system?**
"A few honest ones: retrieval quality caps everything — if the right chunk isn't retrieved, the
LLM can't answer. My AST chunking is Python-first, with a line-window fallback for other
languages. The agent adds latency and cost. And the LLM can still occasionally hallucinate
despite grounding. I mitigate with hybrid search, reranking, citations, and a faithfulness
check, and I *measure* the residual with my eval harness."

**Q: How would you improve it with more time?**
"Swap the fallback chunker for **tree-sitter** to get real AST chunking across all languages;
add **query rewriting** to handle vague questions; add a **'no answer' guardrail** tuned on the
eval set; expand the golden set and track metrics over time; and add **conversation memory** so
follow-up questions keep context."

**Q: What happens if retrieval returns nothing relevant?**
"The prompt instructs the model to reply 'I couldn't find that in the provided sources' rather
than guess — I'd rather it admit ignorance than hallucinate. My golden set has an unanswerable
question that specifically tests this behavior."

**Q: Why Groq / ChromaDB / bge-small specifically?**
"They're all free, local-or-generous, and swappable. Groq gives a free, fast LLM; ChromaDB is
an embedded vector DB with zero infra; bge-small is a small, strong open-source embedding model.
Critically, I put the LLM and retriever behind **interfaces**, so I can swap any of them —
Groq→Claude, ChromaDB→Qdrant — by changing one line."

---

## F. The 60-second pitch (open with this)

> "I built an Agentic Codebase & Document Intelligence Engine — a RAG system that answers
> questions about any repo or docs with source citations. It chunks code by AST, embeds into a
> vector database, and retrieves with hybrid search (semantic + BM25, fused via Reciprocal Rank
> Fusion) plus a cross-encoder reranker. An LLM generates a cited, grounded answer, and an
> agentic loop decomposes hard questions and self-checks the result. I validate everything with
> an evaluation harness measuring context recall, answer correctness, and LLM-judged
> faithfulness. The LLM and retriever are behind swappable interfaces."

Then say: *"Happy to go deep on any part."* — and use the phase docs as your deep-dive.

---

## 🧠 Practice tip
Cover the answers, read each question aloud, and answer in your own words. If you can do that
for sections A–E, you're placement-ready on this project.

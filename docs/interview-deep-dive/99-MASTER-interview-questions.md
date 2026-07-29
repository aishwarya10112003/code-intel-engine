# 99 — MASTER Interview Question Bank (Defend the Whole RAG Project)

*The combined, cross-cutting question set for the entire Code-Intel Engine. The per-subsystem files
(01–13) go deep on one topic each; **this** file is what an interviewer actually does — jump across
subsystems, chain "why?" three levels down, probe failure modes, and test whether you can hold the whole
RAG pipeline in your head. Model answers included. Practice out loud.*

> **How to use this file.** Read it after the subsystem files and the
> **[tech-stack cheat sheet](90-TECH-STACK-CHEATSHEET.md)**. Cover the answers, ask yourself the
> question, speak your answer, then check. If any answer surprises you, go back to the linked file.
> Goal: **never go blank, always have the "why," always know the alternative and why not.**

**Cross-reference legend:** `[01]`=architecture/pipeline · `[02]`=chunking · `[03]`=embeddings ·
`[04]`=vector store/ANN · `[05]`=dense · `[06]`=BM25 · `[07]`=hybrid/RRF · `[08]`=reranking ·
`[09]`=generation/citations · `[10]`=agentic loop · `[11]`=LLM abstraction · `[12]`=evaluation ·
`[13]`=UI/deploy · `F1`=vectors/ANN · `F2`=LLMs/prompting · `90`=tech-stack cheat sheet.

---

## §0. The 60-second pitch (memorize the shape, not the words)

> "Code-Intel Engine is an **agentic RAG system that answers architecture questions about any codebase
> or docs — with exact source citations**. You point it at a repo; it chunks each file on *logical*
> boundaries (Python AST for code, headings for docs), embeds the chunks with a local model into a
> ChromaDB vector index, and for a question it retrieves with **both** semantic vector search **and**
> BM25 keyword search, fuses them with **Reciprocal Rank Fusion**, and sharpens the top candidates with
> a **cross-encoder reranker**. An LLM then writes an answer grounded **only** in those sources, citing
> each claim with `[n]`. For hard, multi-part questions an **agent** decomposes the question, gathers
> context for each part, and **self-critiques** faithfulness with an LLM judge, retrying (capped) if the
> answer isn't grounded. And every config is **measured** against a golden set on context recall, answer
> correctness, and faithfulness — so each upgrade is proven, not guessed. I built it **from scratch** in
> Python — no LangChain — to understand the mechanics, with the LLM and retriever behind interfaces so
> they're one-line swaps."

**Three shorter variants:**
- **One-liner:** "A from-scratch, agentic, hybrid-retrieval RAG engine that answers codebase questions
  with cited, grounded answers — and measures its own quality."
- **ML-lead framing:** "Structural chunking → local embeddings + HNSW → dense+BM25 fused with RRF →
  cross-encoder rerank → grounded cited generation → agentic self-critique → golden-set evaluation."
- **Product framing:** "Ask your repo a question in plain English and get a trustworthy answer with
  clickable sources, even for multi-part questions."

---

## §1. "Tell me about this project" — the STAR narrative

**Situation.** Understanding an unfamiliar codebase is slow, and generic "chat with your code" tools
hallucinate — they answer confidently without showing where the answer came from.

**Task.** Build a system that answers architecture questions about *any* repo or docs with **grounded,
cited** answers, robust enough to handle exact identifiers and multi-part questions — and *prove* its
quality rather than assert it. Constraints: from scratch (to learn the mechanics), free/local where
possible, single machine.

**Action.** I built a RAG pipeline end to end: AST/heading **chunking** so units are meaningful; local
**embeddings** + **ChromaDB/HNSW** for semantic search; **BM25** for exact identifiers; **RRF** to fuse
them; a **cross-encoder reranker** for precision; **grounded prompting with citations** and a safe
refusal to fight hallucination; an **agentic loop** (decompose → gather → self-critique → capped retry)
for hard questions; and an **evaluation harness** (context recall, correctness, faithfulness via
LLM-judge) to measure every change. The LLM and retriever sit behind interfaces, so providers/stores are
one-line swaps.

**Result.** A working, demoable engine (CLI + Streamlit) that gives cited answers, handles multi-part
questions via the agent, and reports measured quality per config — so "hybrid helps" and "the reranker
helps" are numbers, not vibes. It runs on one machine with a free LLM key.

**What I learned.** The two biggest levers in RAG are **retrieval quality** (chunking + hybrid + rerank —
if the right chunk isn't retrieved, nothing saves the answer) and **grounding discipline** (prompt +
citations + measured faithfulness). And that *right-sizing* matters — I deliberately avoided LangChain,
tree-sitter, a server vector DB, and paid APIs because the project's goals didn't need them (see `[90]`).

---

## §2. The spine — say this and most answers follow
**"Chunk on meaning → embed into a vector index → retrieve with meaning (dense) AND exact words (BM25),
fuse (RRF) and rerank (cross-encoder) → generate an answer grounded ONLY in those sources with citations
→ for hard questions, an agent decomposes and self-checks faithfulness → and it's all measured against a
golden set."**

**Q. Why is that the backbone?** "Because it names every stage and the *reason* each exists: meaning
needs semantic search, exact identifiers need keyword search, incompatible scores need RRF, precision
needs a reranker, trust needs grounding+citations, hard questions need an agent, and confidence needs
measurement. State the spine and the interviewer's follow-ups are already answered."

---

## §3. RAG system design

**Q. Design a system to answer questions over a large codebase.** "Offline: chunk each file on logical
boundaries `[02]`, embed the chunks `[03]`, store vectors in an ANN index `[04]`. Online: embed the
query, retrieve with hybrid dense+BM25 `[05][06][07]`, rerank the top candidates `[08]`, build a
grounded prompt with numbered sources, and generate a cited answer `[09]`. Add an agent for multi-part
questions `[10]` and an eval harness to tune it `[12]`. Keep the LLM and store behind interfaces for
swappability `[11][01]`."

**Q. What are the failure points and how do you defend each?**
| Failure | Defense | File |
|---|---|---|
| Right chunk never retrieved | AST chunking + hybrid + rerank; measure context recall | `[02][07][08][12]` |
| Exact identifier missed | BM25 keyword leg | `[06]` |
| Incompatible score scales | RRF (rank-based fusion) | `[07]` |
| Mediocre chunk in top-k | cross-encoder rerank | `[08]` |
| Hallucination | grounded prompt + citations + refusal + faithfulness judge | `[09][10][12]` |
| Multi-part question | agentic decomposition | `[10]` |
| Vendor lock-in | LLMClient/Retriever interfaces | `[11][01]` |
| Ephemeral host wipes index | rebuild from chunks.json on boot | `[04][13]` |

**Q. How does it scale to a huge repo?** "Chunking and embedding are offline and parallelizable; ANN
(HNSW) keeps query time ~log N `[F1][04]`. The bottleneck at real scale is the single-machine embedded
Chroma — I'd swap to Qdrant/pgvector behind the same `VectorStore` interface `[04]`. BM25 in memory would
move to a served index (e.g., Elasticsearch) if the corpus outgrew RAM `[06]`. The pipeline and prompts
don't change — that's the interface design paying off `[01]`."

**Q. Where's the latency, and how do you manage it?** "Model loading (cached once, `[13]`), the
cross-encoder (bounded to ~15 candidates, `[08]`), the agentic loop (capped retries + ≤3 sub-questions,
`[10]`), and the LLM call itself (Groq is fast). Simple queries can skip rerank/agent for speed."

---

## §4. The "why not X?" gauntlet (the cheat sheet, spoken)

*(Full tables in [`90-TECH-STACK-CHEATSHEET.md`]. The ones you must know cold:)*

| If they ask… | One-line answer | File |
|---|---|---|
| Why not LangChain/LlamaIndex? | "From scratch to own the mechanics; frameworks hide the control flow." | `[01][10]` |
| Why not fixed-size chunks? | "They cut functions in half — you retrieve fragments." | `[02]` |
| Why not OpenAI embeddings? | "Cost + data egress; local bge-small is free, private, swappable." | `[03]` |
| Why not brute-force search? | "O(N) per query; HNSW is ~log N." | `[04][F1]` |
| Why hybrid, isn't semantic enough? | "Semantic misses exact identifiers; BM25 nails them." | `[07]` |
| Why RRF, not add scores? | "Scales are incomparable; RRF fuses on rank." | `[07]` |
| Why a reranker? | "Retrieval is fast but blurry; a cross-encoder is precise on the top few." | `[08]` |
| Why Groq, not GPT-4/Claude? | "Free, fast, open — and one-line swappable behind the interface." | `[11]` |
| Why temperature 0.1? | "Faithful, repeatable answers, not creativity." | `[09]` |
| Why build the agent yourself? | "To own the decompose/critique/retry loop; ~90 lines, hard cap." | `[10]` |
| Why LLM-as-judge? | "Faithfulness is semantic; rules can't measure it; runs offline." | `[12]` |
| Why Streamlit? | "Fastest Python-function-to-demo; a React+API stack adds nothing here." | `[13]` |

**Q. Defend your most controversial 'not' — no LangChain.** "LangChain would let me ship faster, but the
*point* of this project is to demonstrate I understand production RAG — chunking, RRF, two-stage
retrieval, the agent loop. A framework abstracts exactly those away and adds heavy dependencies and
hidden control flow. I hand-wrote each in a few hundred defensible lines, and I kept `Retriever`/
`LLMClient` interfaces so I could adopt a framework later without a rewrite. That's owning the mechanics
without closing the door."

---

## §5. Retrieval deep-questions (the heart of RAG)

**Q. Dense vs sparse — why both?** "Dense embeddings match *meaning* (paraphrases) but under-rank exact
tokens; sparse BM25 matches *exact words* (identifiers, error codes) but misses paraphrases. Code needs
both, so I fuse them `[05][06][07]`."

**Q. Walk RRF with numbers.** "Each list contributes `1/(k+rank)`, k=60. A chunk at rank 1 in dense and
rank 3 in BM25 scores `1/61 + 1/63`; a chunk at rank 2 in both scores `1/62 + 1/62` — the one in *both*
lists can win, because RRF rewards cross-retriever agreement. And a chunk only one retriever found still
contributes and can make the top-k. Rank-based, so the two incompatible score scales never touch `[07]`."

**Q. Bi-encoder vs cross-encoder?** "Bi-encoder embeds query and chunk separately → fast, scalable,
blurry (retrieval). Cross-encoder reads them together → accurate, slow (reranking). Two-stage: retrieve
fast to ~15 candidates, then cross-encode those to the best 5. Precision where the LLM sees it, without
paying the cost over the whole corpus `[08]`."

**Q. What's the single biggest lever on answer quality?** "Retrieval — specifically context recall. If
the answer-bearing chunk isn't retrieved, no prompt or model recovers it. That's why chunking, hybrid,
and rerank get the most attention, and why the eval harness measures recall first `[02][07][08][12]`."

**Q. Why does chunking matter so much?** "It caps everything downstream. Fixed-size chunks cut a function
in half, so you retrieve and feed the LLM a fragment. AST/heading chunks are self-contained units, so a
retrieved chunk is actually usable. It's the highest-leverage single decision in the system `[02]`."

---

## §6. Generation, grounding & hallucination

**Q. How do you stop it hallucinating?** "Layered. Prompt: use ONLY the numbered sources, cite every
claim `[n]`, refuse with a fixed sentence if the answer's absent, never invent — at temperature 0.1.
Agent: an LLM faithfulness judge can trigger a retry. Eval: faithfulness is measured on a golden set. So
grounding is enforced, caught, *and* measured — and I'm honest that the prompt alone isn't a guarantee
`[09][10][12]`."

**Q. Why citations specifically?** "They force attribution — every claim must point at a source — and let
the user verify. An uncited sentence is a visible red flag. Returning the sources with the answer makes
the whole thing checkable `[09]`."

**Q. What is RAG vs fine-tuning, and why RAG here?** "Fine-tuning changes weights — good for style/format,
expensive, and still wouldn't know *this* repo. RAG injects current source text at query time and cites
it. For 'answer questions about a specific, changing codebase', RAG is correct; I'd only fine-tune for a
fixed output format `[F2][09]`."

---

## §7. The agent & evaluation

**Q. What makes it 'agentic', concretely?** "The program decides how to solve the task: an LLM decomposes
the question into sub-questions, retrieves for each, generates, and an LLM judge decides whether the
answer is faithful enough — if not, it gathers more context and retries, up to a hard cap. Decisions and
control flow driven by model judgments, not a fixed path `[10]`."

**Q. How do you prevent an infinite agent loop?** "A hard `max_retries` cap (default 1) and a
sub-question cap (3). The retry condition needs both a low faithfulness score and attempts under the cap,
so termination is guaranteed by construction `[10]`."

**Q. How do you know any of this actually works?** "The eval harness. A golden set scored on context
recall, answer correctness, and faithfulness (LLM-judge), per config. I can prove dense→hybrid→rerank
helps and catch regressions. I also verify the metrics *fail* for the right reasons — break retrieval and
recall drops; loosen grounding and faithfulness drops `[12]`."

**Q. Aren't near-perfect eval scores suspicious?** "On a small clean corpus, yes, and I say so. The
harness's value is regression-catching and tuning on larger, noisier corpora, not the absolute number on
a toy set `[12]`."

---

## §8. Failure modes — "what happens if…" rapid drill

| Scenario | What happens | File |
|---|---|---|
| A Python file won't parse | AST chunker returns empty → line-window fallback; file still indexed | `[02]` |
| Query is an exact identifier | BM25 leg matches it; RRF keeps it | `[06][07]` |
| Dense and BM25 disagree | RRF blends by rank; reranker sharpens final order | `[07][08]` |
| Best chunk is at rank 8 | cross-encoder promotes it into top-5 | `[08]` |
| Answer isn't in the sources | fixed refusal "I couldn't find that…" (tested) | `[09][12]` |
| Multi-part question | agent decomposes into sub-questions | `[10]` |
| Answer is unfaithful | judge scores low → agent gathers more + retries (capped) | `[10]` |
| Judge returns junk | regex-extract 1–5, default 3 → eval never crashes | `[12]` |
| Redis/index empty | pipeline returns "run build_index.py first" | `[01]` |
| Ephemeral host wiped `.chroma/` | `ensure_index` rebuilds from chunks.json on boot | `[04][13]` |
| Missing API key | clear error with signup URL; UI shows st.error | `[11][13]` |
| Swap LLM provider | new adapter + one `elif` + env var | `[11]` |
| Corpus grows huge | swap Chroma→Qdrant behind the interface; HNSW keeps queries ~log N | `[04][F1]` |

---

## §9. Behavioral / reflective

**Q. What was the hardest part?** "Getting *retrieval* right — realizing dense alone misses exact
identifiers, that fusing scores directly is meaningless, and that the fix is RRF plus a two-stage
reranker. The insight was that answer quality is dominated by whether the right chunk reaches the LLM,
so I invested there and measured it with context recall `[07][08][12]`."

**Q. What would you do next?** "tree-sitter for multi-language AST chunking `[02]`; query rewriting for
vague questions and conversation memory for follow-ups `[10]`; swap embedded Chroma → Qdrant/pgvector for
scale `[04]`; and a bigger, noisier eval set with coverage reporting `[12]`. All are on the README
roadmap."

**Q. What are you most proud of?** "That every claim is *provable* and *honest*. 'Hybrid helps' is a
number from the eval harness, not a vibe. And I state the limits plainly — Python-only AST, small golden
set, single-machine store — because knowing the edges is part of engineering `[12][90]`."

**Q. Where did you deliberately NOT gold-plate?** "No LangChain, no tree-sitter yet, no server vector DB,
no paid APIs, no React frontend. Each rejected with a scale/goal-specific reason `[90]`. Engineering
maturity is saying no to complexity that doesn't earn its keep."

**Q. Did you use AI to build it? Own it.** "I used modern tooling and can defend every line — that's what
these notes are for. Ask me why any component, parameter, or trade-off exists and I'll give the reason,
the alternative, and why I rejected it."

---

## §10. Rapid-fire one-liners (know these cold)

- **Why chunk on AST?** Meaningful units; fixed-size cuts functions in half. `[02]`
- **Why a local embedding model?** Free, private, fast, swappable. `[03]`
- **Why normalize embeddings?** Cosine = dot product; no length bias. `[F1][03]`
- **Why ChromaDB?** Embedded, zero-ops, persists locally. `[04]`
- **Why HNSW?** ~log-N approximate search; brute force is O(N). `[F1][04]`
- **Why BM25?** Exact identifiers/error codes semantic search misses. `[06]`
- **Why RRF?** Fuse incompatible score scales by rank. `[07]`
- **Why a cross-encoder rerank?** Precision on the top candidates the LLM sees. `[08]`
- **Why citations + "only the sources"?** Grounding + verifiability vs hallucination. `[09]`
- **Why an agent?** Multi-part questions + self-checked faithfulness. `[10]`
- **Why interfaces for LLM/retriever?** One-line provider/store swaps. `[11][01]`
- **Why LLM-as-judge?** Faithfulness is semantic; runs offline. `[12]`
- **Why a golden set?** Prove upgrades, catch regressions. `[12]`
- **Why Streamlit + cache_resource?** Fast demo; load heavy models once. `[13]`
- **Why ship chunks.json not .chroma/?** Cheap source artifact; rebuild the derived index. `[04][13]`

---

## §11. Curveballs & stress questions

**Q. Isn't 'from scratch' just avoiding LangChain to look clever?** "It's a deliberate learning choice
with a real payoff: I can explain and control every stage, and the interfaces mean I'm not locked out of
frameworks later. For a portfolio project whose goal is *demonstrating* RAG understanding, owning the
mechanics is the feature `[01][10]`."

**Q. Your eval scores are 1.0 — did you overfit to the golden set?** "The golden set is tiny and clean,
so high scores are expected — I report that caveat explicitly. The harness's job is catching regressions
and tuning on larger corpora, not proving perfection on six questions `[12]`."

**Q. Semantic search is 'AI'; BM25 is 'old'. Why bother with BM25?** "Because 'old' here means 'reliable
and exactly right for exact tokens'. Code is full of identifiers where lexical match beats embeddings.
Combining a decades-proven lexical method with modern semantics is *stronger* than either alone — that's
the whole hybrid thesis `[06][07]`."

**Q. If retrieval is everything, why the agent and reranker?** "Retrieval quality is necessary but not
sufficient. The reranker improves *ordering* of what retrieval found (precision at top-k), and the agent
handles *question shape* (multi-part) and *output verification* (faithfulness). They attack different
failure modes than raw recall `[08][10]`."

**Q. Where does this break first in production?** "The single-machine embedded store and in-memory BM25 —
both are RAM/one-box bound. The fix is swapping to a served vector DB and search index behind the
existing interfaces. Also latency under the agent + reranker; I'd make those optional per query. I know
exactly where the edges are `[04][06][08][10]`."

**Q. Why should I trust the answers at all?** "Because they're grounded in retrieved sources, every claim
is cited, the system refuses when it doesn't know, and faithfulness is *measured*. Trust isn't asserted —
it's built from citations you can click and a metric you can check `[09][12]`."

---

## §12. Smart questions to ask THE INTERVIEWER

- "For your RAG, where did you land on chunking — structural vs fixed-size vs semantic?"
- "Do you use hybrid retrieval, and how do you fuse — RRF, weighted, or a learned ranker?"
- "How do you evaluate RAG changes — golden sets, LLM-judge, human eval, or online metrics?"
- "How do you handle hallucination and grounding guarantees in production?"
- "Where's your trust boundary — how much do you let the model decide vs constrain it?"

---

## §13. 10-minute pre-interview warm-up checklist

1. Say the **60-second pitch** out loud (§0).
2. Recite the **spine** and why each stage exists (§2).
3. Explain **dense vs BM25 vs RRF vs rerank** as four distinct jobs (§5).
4. Walk **RRF with numbers** (§5).
5. Explain **grounding + citations + refusal + faithfulness judge** as layered anti-hallucination (§6).
6. Describe the **agent loop** and why the retry cap matters (§7).
7. Explain the **eval harness** and its three metrics (§7).
8. Defend **3 "why not X"** picks cold — LangChain, OpenAI embeddings, brute-force search (§4, `[90]`).
9. Name the **honest limitations** and your fixes (§9, `[90]`).
10. Breathe. You built this. Every answer traces to a file in this folder.

---

⬅️ Back to [`00-INDEX.md`](00-INDEX.md) · cheat sheet: [`90-TECH-STACK-CHEATSHEET.md`](90-TECH-STACK-CHEATSHEET.md) ·
deep-dives `[01]`–`[13]` · fundamentals `F1` `F2`

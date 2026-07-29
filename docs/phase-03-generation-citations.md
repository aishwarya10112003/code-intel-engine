# Phase 3 — Generation with Citations (The First Full RAG Answer)

## 🎯 What this phase does
Ties everything together into a real question-answering system: you ask a question, it
retrieves the relevant chunks (Phase 2), hands them to a language model, and gets back a
written answer that **cites its sources**. This is the first moment the whole thing behaves
like the product we set out to build.

---

## 🧠 The big idea first: the complete RAG loop

RAG = **Retrieve, Augment, Generate**. Phase 2 gave us Retrieve. Now we add the other two:

```
   Your question
        │
        ▼
   1. RETRIEVE  → embed the question, get the top-k most relevant chunks   (Phase 2)
        │
        ▼
   2. AUGMENT   → paste those chunks into the prompt as numbered "sources" [1] [2] [3]
        │
        ▼
   3. GENERATE  → the LLM writes an answer using ONLY those sources, citing them by number
        │
        ▼
   Answer + a list mapping [1][2][3] back to real files
```

**Why this matters:** the LLM has never seen your private code. Instead of hoping it
"knows," we *hand it the exact relevant text* and ask it to answer from that. This is what
makes an LLM useful on private data — and citations make the answer **verifiable**.

---

## 🔩 What we built (the files)

```
src/llm/base.py        → LLMClient interface (a swappable "talk to a model" contract)
src/llm/groq_client.py → Groq adapter (free, fast Llama models)
src/llm/__init__.py     → factory: picks the provider from an env var
src/rag/pipeline.py     → the RAG orchestration (retrieve → augment → generate)
ask.py                  → the command you run: question → cited answer
```

## 1. The swappable LLM layer (great engineering + great interview point)

We did **not** hard-wire Groq into the RAG logic. Instead we defined a tiny interface —
`LLMClient` with one method, `generate(system, user)` — and made Groq *one implementation*
of it. A factory (`get_llm()`) picks the provider from the `LLM_PROVIDER` environment
variable.

**Why bother?** Because the model provider is a detail that shouldn't leak into the core
logic. Today it's Groq (free); tomorrow you could add Gemini, a local Ollama model, or
Claude by writing one small adapter and one `elif` — the RAG pipeline never changes. This is
the **"program to an interface, not an implementation"** principle, and it's exactly the
kind of design decision interviewers love.

## 2. The prompt — where hallucination is fought

The **system prompt** is the model's rulebook. Ours enforces four anti-hallucination rules:
- Answer using **only** the provided numbered sources — no outside knowledge.
- **Cite** every claim with its source number, e.g. `[1]`, `[2]`.
- If the answer isn't in the sources, say so — don't guess.
- Never invent files or functions.

We set **temperature = 0.1** (low). Temperature controls randomness/creativity; for a
factual assistant we want it near zero so the model sticks to the sources.

## 3. The citation mechanism

In `pipeline.py` we number the retrieved chunks `[1] file: app.py (deposit) ...`, `[2] ...`
and put them in the prompt. The model cites `[1]`, `[2]` in its answer. Then `ask.py` prints
a **Sources** list mapping each number back to the real chunk id and file. Result: every
sentence in the answer can be traced to an exact location in your codebase. That
traceability is the whole point of a *trustworthy* code assistant.

## 4. How to run it (once your free key is set)

```bash
export GROQ_API_KEY="gsk_...your free key..."     # from console.groq.com
python ingest.py sample_input          # (Phase 1) chunks
python build_index.py chunks.json      # (Phase 2) index
python ask.py "how do I deposit money into an account"   # (Phase 3) cited answer
```

You'll get something like:
```
=== ANSWER ===
To add money, call the deposit method, which increases the balance after checking the
amount is positive [1].

=== SOURCES ===
  [1] example.py::BankAccount.deposit   (similarity 0.78)
```

---

## 🔑 Words you must know (this phase)

- **RAG loop** — Retrieve → Augment → Generate.
- **Prompt** — the full text sent to the LLM (system rules + user message + sources).
- **System prompt** — the instructions that set the model's behavior/rules.
- **Grounding** — forcing the model to answer only from provided sources.
- **Hallucination** — when an LLM confidently makes something up; grounding + citations fight it.
- **Citation** — a reference `[n]` linking a claim to its exact source.
- **Temperature** — randomness knob; low = factual, high = creative.
- **Interface / adapter** — the swappable-provider design (LLMClient + GroqClient).

---

## 🛡️ Interview defense (say these out loud)

> *"How does your system answer questions about code the model was never trained on?"*
> "It's **RAG**. I retrieve the most relevant chunks by semantic search, inject them into
> the prompt as numbered sources, and instruct the model to answer using **only** those
> sources and cite them. The model isn't recalling my code from training — it's reasoning
> over text I hand it at query time."

> *"How do you stop it from hallucinating?"*
> "Several layers: a strict system prompt that forbids outside knowledge and requires a
> 'not found' answer when the sources don't cover it, **low temperature** for factual output,
> and mandatory **citations** so every claim is traceable to a real file. In Phase 7 I add
> an automated grounding check on top."

> *"Why abstract the LLM behind an interface?"*
> "So the provider is a swappable detail. My RAG logic depends on a one-method `LLMClient`
> interface; Groq is just one adapter. I can switch to Gemini, a local model, or Claude by
> adding an adapter — the core never changes. Program to an interface, not an implementation."

**Keywords to drop:** *RAG, grounding, retrieval-augmented, system prompt, hallucination
mitigation, citations/traceability, temperature, provider abstraction / adapter pattern.*

---

## ✅ What you can now say you built
1. A complete **RAG loop**: retrieve → augment → generate.
2. A **swappable LLM layer** (interface + Groq adapter + factory).
3. A grounded, **citation-backed** answering command (`ask.py`) with anti-hallucination prompting.

➡️ Next (Phase 4): an **evaluation harness** — so we can measure answer quality with numbers
and prove that every future upgrade actually helps.

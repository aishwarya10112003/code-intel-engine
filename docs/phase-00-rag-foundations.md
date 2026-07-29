# Phase 0 — RAG From Absolute Zero (Read This First)

> This assumes you know **nothing** about LLMs or RAG. By the end you'll understand every
> foundational term the rest of the docs use. Read slowly; it's the base everything sits on.

---

## 1. What is an LLM (Large Language Model)?

An **LLM** is a very large neural network (like the model behind ChatGPT or Claude) trained on
enormous amounts of text. Its one core skill sounds simple but is powerful: **given some text,
predict what comes next.** By doing that extremely well, it can answer questions, write code,
summarize, translate, and so on.

Two things you must know about LLMs for interviews:

- **It only knows what it was trained on, up to a cutoff date.** It has *never seen your
  private code* or your company's docs. Ask it about your codebase and it will either say "I
  don't know" or — worse — **make something up**.
- **It has a limited "context window".** The context window is the maximum amount of text the
  model can read in one go (measured in *tokens* — see below). You cannot paste a whole
  codebase into it.

These two limits are *exactly* what RAG exists to solve.

## 2. What is a token?

LLMs don't read words or characters — they read **tokens**, which are chunks of text roughly
¾ of a word on average (e.g. "deposit" might be one token, "unbelievable" might be three:
"un", "believ", "able"). Everything is measured in tokens: the context window, the price, the
limits. *"This model has a 128k context window"* means it can read ~128,000 tokens (~100k
words) at once.

**Why you care:** you pay per token, and you can only fit so many tokens. That's *why* RAG
sends the LLM only the few most relevant chunks instead of everything.

## 3. What is an embedding? (the heart of RAG)

An **embedding** is a list of numbers (a **vector**) that represents the *meaning* of a piece
of text. A separate, smaller neural network (an **embedding model**) produces it. It's trained
so that **texts with similar meaning get vectors that are close together**, and different
meanings get vectors that are far apart.

Analogy: imagine every sentence becomes a dot on a giant map. "How do I add money" and
"deposit funds" land near each other; "Kubernetes deployment" lands far away. Once text is
dots on a map, *"find the most relevant text"* becomes *"find the nearest dots"* — just
distance math.

- Our embedding model outputs **384 numbers** per text (its "dimension").
- We measure closeness with **cosine similarity** (1.0 = identical meaning, 0 = unrelated).

**This is the single most important concept in the whole project.** If you understand
embeddings, you understand RAG.

## 4. What is RAG, and why does it exist?

**RAG = Retrieval-Augmented Generation.** It's the standard technique for making an LLM answer
questions about data it was never trained on (your code, your docs).

The insight: **don't rely on the LLM's memory — give it the relevant text at question time.**

```
   1. RETRIEVE  find the chunks of YOUR data most relevant to the question (via embeddings)
   2. AUGMENT   paste those chunks into the prompt as "context"
   3. GENERATE  ask the LLM to answer using ONLY that context (and cite it)
```

This solves both LLM limits from section 1: it *doesn't need to have memorized* your data
(retrieval provides it), and it *doesn't need the whole codebase in context* (only the few
relevant chunks go in).

## 5. What is "hallucination"?

**Hallucination** is when an LLM confidently states something false — an invented function, a
wrong file, a made-up fact. It happens because the model is *predicting plausible text*, not
looking up truth. RAG fights hallucination two ways: (1) grounding — the model answers from
retrieved real text, and (2) **citations** — every claim points to a source, so it's
verifiable and the model is discouraged from inventing.

## 6. The question you WILL be asked: RAG vs Fine-tuning

There are two ways to make an LLM "know" your data. Know the difference cold:

| | **RAG** (what we built) | **Fine-tuning** |
|---|---|---|
| What it does | Retrieves your data at question time and puts it in the prompt | Re-trains the model's weights on your data |
| Updating data | Instant — just re-index. New doc? It's searchable immediately. | Expensive — you must retrain to add/change data |
| Cost | Cheap, no training | Expensive, needs GPUs + ML expertise |
| Citations | Yes — you know exactly which source was used | No — the knowledge is baked in, untraceable |
| Best for | Knowledge that changes, and needing traceable sources | Teaching the model a *style* or *skill*, not facts |

**One-liner for the interview:** *"RAG injects knowledge at query time and is ideal for
changing, citable facts; fine-tuning bakes knowledge into the weights and is better for
teaching a style or behavior. For a codebase Q&A tool — data that changes and needs
citations — RAG is clearly the right choice."*

## 7. The vocabulary map (how the pieces connect)

```
   Your files
      │  chunking (split into pieces)
      ▼
   Chunks ──embedding model──►  Vectors (meaning as numbers)
                                   │  stored in a
                                   ▼
                              Vector Database
                                   │  a Question is also embedded → find nearest vectors
                                   ▼
                          Retrieved chunks (the context)
                                   │  put in the prompt (augment)
                                   ▼
                                  LLM ──►  Cited answer (generation)
```

---

## 🛡️ Foundational interview answers (say these out loud)

> *"What's an embedding?"* — "A vector of numbers representing a text's meaning, made by an
> embedding model trained so similar meanings are close together. It turns 'find relevant
> text' into 'find nearby vectors'."

> *"Why do you need RAG — can't the LLM just answer?"* — "The LLM was never trained on my
> private code and can't fit a whole codebase in its context window. RAG retrieves the
> relevant chunks and hands them to the model at query time, so it answers from real, current
> data instead of guessing."

> *"RAG or fine-tuning?"* — [use the one-liner in section 6].

> *"What's hallucination and how do you handle it?"* — "It's when the model confidently makes
> something up, because it predicts plausible text rather than looking up truth. I fight it
> with grounding — answering only from retrieved sources — plus mandatory citations and a
> faithfulness check."

**Keywords:** *LLM, token, context window, embedding, vector, cosine similarity, retrieval,
grounding, hallucination, RAG vs fine-tuning.*

➡️ Now read [phase-01-chunking.md](phase-01-chunking.md) and go through the phases in order.

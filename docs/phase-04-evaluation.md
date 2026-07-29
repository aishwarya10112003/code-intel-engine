# Phase 4 — Evaluation (Measuring Quality With Numbers)

## 🎯 What this phase does
Builds a system that **grades our RAG engine automatically**. It runs a fixed set of test
questions with known-good answers (a "golden set") and produces scores. From now on, every
upgrade must *prove* it helped by moving these numbers.

## 🧠 Why this is the most important phase for your resume
Anyone can wire up a RAG demo. Almost nobody measures it. If you can say *"reranking
improved my faithfulness from 0.78 to 0.91 on my golden set,"* you sound like an engineer,
not a tutorial-follower. **Evaluation is the difference.** We build it *before* the fancy
retrieval upgrades so each one gets measured.

---

## 🔩 What we built

```
tests/golden.json      → the test questions + expected facts (the "answer key")
src/eval/judge.py       → LLM-as-judge: grades faithfulness (is the answer grounded?)
evaluate.py             → runs the golden set through a config and prints scores
```
Plus a small **refactor**: retrieval now lives behind a `Retriever` interface, and a
`build_pipeline("dense" | "hybrid" | "hybrid_rerank")` factory. That's what lets us evaluate
different setups with one command.

## The three metrics (what each measures, in plain words)

1. **Context Recall** — *"Did we even retrieve the chunk that holds the answer?"* If the
   right chunk never makes it into the top-k, the LLM has no chance. We check whether the
   expected chunk id appears in what was retrieved. (This measures the *retriever*.)

2. **Answer Correctness** — *"Does the final answer contain the key fact?"* A simple, fast
   check: does the answer include the expected keyword(s) (e.g. `STRIPE_SECRET_KEY`)? (This
   measures the *whole pipeline*.)

3. **Faithfulness** — *"Is the answer actually supported by the sources, or did it make
   something up?"* This is the anti-hallucination score. It's hard to check with string
   rules, so we use an **LLM-as-judge**: we show another LLM call the question, our answer,
   and the sources, and ask it to rate support 1–5. (This measures *trustworthiness*.)

## LLM-as-judge — a key technique to name in interviews

Using an LLM to grade another LLM's output is a standard evaluation method. It works because
judging "is this answer supported by this text?" is *easier* than producing the answer, and
an LLM does it consistently. Crucially it runs **offline** (only during evaluation), so its
extra cost and latency never affect real users.

## Our baseline result (real output)

```
SCORES for 'dense':
  Context Recall     : 1.00   (5/5)
  Answer Correctness : 1.00   (6/6)
  Faithfulness       : 0.87   (avg 4.3/5)
```

**Honest note:** our sample corpus is tiny (11 chunks), so plain vector search already nails
recall. On a small, clean corpus that's expected. The harness's value grows with the
codebase: on a real 10,000-chunk repo, these numbers are how you'd *tune* hybrid search and
reranking and catch regressions. The harness is the instrument; the corpus decides how much
headroom there is to improve.

---

## 🔑 Words you must know
- **Golden set / test set** — fixed questions with known-good answers; your "answer key".
- **Evaluation harness** — the script that runs the golden set and scores the system.
- **Context Recall** — did retrieval fetch the chunk containing the answer?
- **Answer Correctness** — does the answer contain the expected fact?
- **Faithfulness** — is the answer supported by its sources (no hallucination)?
- **LLM-as-judge** — using an LLM to grade outputs.
- **Regression** — a change that makes a metric worse; the harness catches these.
- **Retriever interface** — the swappable abstraction that lets us evaluate configs.

---

## 🛡️ Interview defense
> *"How do you know your RAG system is any good?"*
> "I don't guess — I measure. I built an **evaluation harness** with a golden test set and
> three metrics: **context recall** (did we retrieve the right chunk), **answer
> correctness** (did the answer contain the key fact), and **faithfulness** (is the answer
> supported by its sources, graded by an **LLM-as-judge**). Every retrieval change is
> validated against these numbers, so I can prove an upgrade helped and catch regressions."

> *"Isn't using an LLM to grade an LLM circular?"*
> "Judging is easier than generating — checking whether an answer is supported by given text
> is a simpler, more reliable task than producing the answer. It runs offline during
> evaluation only, so it doesn't affect production latency, and I pair it with deterministic
> checks (recall, keyword correctness) so I'm not relying on the judge alone."

**Keywords:** *golden dataset, evaluation harness, context recall, faithfulness, LLM-as-judge,
regression testing, offline evaluation, metric-driven development.*

---

## ✅ What you can now say you built
1. A **golden test set** and an **evaluation harness** producing three normalized metrics.
2. An **LLM-as-judge** for faithfulness (hallucination detection).
3. A **Retriever interface + pipeline factory** so any config can be measured with one command.
4. A recorded **baseline** to compare every future upgrade against.

➡️ Next (Phase 5): **hybrid search** — combine keyword + semantic retrieval and measure the change.

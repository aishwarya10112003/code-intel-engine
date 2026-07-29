# Phase 7 — The Agentic Loop (Plan, Gather, Answer, Self-Check)

## 🎯 What this phase does
Upgrades the system from "one search → one answer" into an **agent** that plans its approach
and checks its own work: it breaks a complex question into parts, gathers evidence for each,
writes an answer, and then **verifies the answer is actually supported** — retrying if not.

## 🧠 The big idea: what makes it "agentic"

**Agentic** = the program makes its own decisions during a task, instead of following one
fixed script. Ours makes two kinds of decisions:
1. **How to split the question** (query decomposition).
2. **Whether the answer is good enough, or needs another try** (self-critique).

This directly fixes two weaknesses of plain RAG:
- **Multi-part questions.** "How do I add money **AND** what stops over-withdrawing?" — one
  search can't cover both well. The agent splits it and searches each part.
- **Silent hallucination.** Plain RAG never checks itself. The agent grades its own answer
  and retries with more context if it's weak.

---

## 🔩 What we built

```
src/agent/agent.py   → the AgenticRAG orchestrator (decompose → gather → generate → critique → retry)
ask_agent.py          → the command to run the agentic path
```

## The loop, step by step

```
   Complex question
        │
   1. DECOMPOSE   LLM splits it into 1-3 focused sub-questions
        │
   2. GATHER      retrieve for EACH sub-question; pool the unique chunks
        │
   3. GENERATE    write an answer from the pooled sources (cited)
        │
   4. CRITIQUE    an LLM "critic" scores faithfulness (is it supported?)
        │
        ├── score good enough  → done ✓
        └── score too low      → gather MORE context, regenerate  (up to a hard cap)
```

**Real run** — for *"how do I add money, and what prevents over-withdrawing?"* the agent:
- split it into 3 sub-questions,
- gathered 8 unique chunks,
- answered **both parts** correctly with citations,
- self-check faithfulness = **5/5**, so **0 retries** needed.

## Two safety details that matter (and impress interviewers)

- **A hard retry cap (`max_retries`).** An agent that retries forever is a bug and a bill.
  Ours can retry at most a fixed number of times, then returns its best answer. Always cap
  agent loops.
- **The self-critic reuses the same faithfulness judge from Phase 4.** The grounding check
  isn't hand-wavy — it's the exact metric we evaluate with, now used *live* as a quality gate.

## Cost/latency honesty
The agent makes several LLM calls (decompose + generate + critique + maybe a retry), so it's
slower and pricier than plain `ask.py`. That's the deliberate trade: use the fast path for
simple questions, the agent for hard, multi-part ones. Knowing *when* to spend the extra
calls is itself good engineering judgement.

---

## 🔑 Words you must know
- **Agent / agentic** — a program that decides its own steps during a task.
- **Query decomposition** — splitting a complex question into focused sub-questions.
- **Multi-hop question** — one needing evidence from several places.
- **Self-critique / grounding check** — the system grading its own answer's support.
- **Retry cap** — a hard limit so the loop can't run forever.
- **Fast path vs agent path** — cheap single-shot vs thorough multi-step answering.

---

## 🛡️ Interview defense
> *"What's 'agentic' about your RAG system?"*
> "It makes decisions instead of following one fixed path. For a complex question it does
> **query decomposition** — an LLM splits it into sub-questions — retrieves for each, then
> after answering it runs a **self-critique**: an LLM grades whether the answer is actually
> supported by the sources, and if it's weak the agent gathers more context and retries. It's
> plan → gather → answer → verify, with a **hard retry cap** so it can't loop forever."

> *"How do you stop the agent from hallucinating or spinning?"*
> "The grounding critic is the same **faithfulness judge** from my evaluation harness, used
> live as a quality gate — so 'is this supported?' is measured, not assumed. And every retry
> is capped, so worst case it returns its best grounded attempt. I reserve the agent for hard
> multi-part questions and use the cheap single-shot path for simple ones."

**Keywords:** *agentic RAG, query decomposition, multi-hop, self-critique / grounding gate,
LLM-as-judge in the loop, bounded retries, plan-act-verify.*

---

## ✅ What you can now say you built
1. An **agentic loop**: decompose → gather → generate → self-critique → bounded retry.
2. **Query decomposition** for multi-part questions.
3. A **live grounding gate** reusing the faithfulness judge, with a hard retry cap.

➡️ Next (Phase 8): a **Streamlit chat UI** and the final project README — the demo you show.

"""
LLM-as-judge — using an LLM to *grade* our system's answers.

WHY:
  "Faithfulness" = is every claim in the answer actually supported by the retrieved
  sources (i.e. no hallucination)? That's hard to check with simple string rules, but easy
  for an LLM: we show it the question, our answer, and the sources, and ask it to score how
  well-supported the answer is. This is a standard, powerful evaluation technique.

  It runs OFFLINE (only when we evaluate), so its extra cost/latency never touches real users.
"""
from __future__ import annotations

import re
from typing import Any

from src.llm import get_llm

_JUDGE_SYSTEM = """You are a strict grader measuring FAITHFULNESS.

You are given a QUESTION, an ANSWER, and the SOURCES the answer was supposed to rely on.
Decide how well every claim in the ANSWER is directly supported by the SOURCES.

Reply with ONLY a single integer from 1 to 5:
5 = every claim is directly supported by the sources; no hallucination
4 = supported, with tiny unsupported wording
3 = partially supported
2 = mostly unsupported
1 = fabricated / contradicts the sources

Output just the number, nothing else."""


def judge_faithfulness(question: str, answer: str, hits: list[dict[str, Any]]) -> int:
    """Return an integer 1-5 for how faithful the answer is to its sources."""
    llm = get_llm()
    sources = "\n\n".join(f"[{i}] {h['content']}" for i, h in enumerate(hits, 1))
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{sources}"
    raw = llm.generate(_JUDGE_SYSTEM, user).strip()
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else 3

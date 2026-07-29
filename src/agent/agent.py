"""
AgenticRAG — the system that plans, gathers, answers, and CHECKS ITSELF.

Plain RAG does one retrieve → one answer. That struggles with:
  * multi-part questions ("how does auth work AND how are passwords stored?") — one search
    can't cover both well, and
  * silent hallucination — nothing verifies the answer is actually grounded.

The agentic loop fixes both with three moves:

  1. DECOMPOSE  — an LLM breaks a complex question into focused sub-questions.
  2. GATHER     — we retrieve for EACH sub-question and pool the unique chunks.
  3. GENERATE + CRITIQUE + RETRY — we answer, then an LLM "critic" checks whether the answer
     is truly supported by the sources. If not, we fetch more context and try again — up to a
     hard cap so it can never loop forever.

"Agentic" = the program makes its own decisions (how to split the question, whether to retry)
instead of following one fixed path.
"""
from __future__ import annotations

from typing import Any

from src.eval.judge import judge_faithfulness
from src.llm import get_llm
from src.rag.pipeline import SYSTEM_PROMPT, RagPipeline, _format_sources

_DECOMPOSE_SYSTEM = """You break a user's question into focused sub-questions for searching a codebase or docs.

- If the question asks about several distinct things, output 2-3 short sub-questions, one per line.
- If it's already a single focused question, output it unchanged on one line.
- Output ONLY the sub-questions, no numbering, no extra text."""


class AgenticRAG:
    def __init__(self, pipeline: RagPipeline, max_retries: int = 1, faithful_threshold: int = 4) -> None:
        self.pipeline = pipeline
        self.llm = get_llm()
        self.max_retries = max_retries          # hard cap — prevents infinite loops
        self.faithful_threshold = faithful_threshold

    # ---- Step 1: decompose ---------------------------------------------------------
    def _decompose(self, question: str) -> list[str]:
        raw = self.llm.generate(_DECOMPOSE_SYSTEM, question)
        subs = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
        # Safety: never return nothing; cap at 3 to control cost.
        return subs[:3] if subs else [question]

    # ---- Step 2: gather unique context ---------------------------------------------
    def _gather(self, questions: list[str], seen: set[str], pool: list[dict[str, Any]]) -> None:
        for q in questions:
            for hit in self.pipeline.retrieve(q):
                if hit["chunk_id"] not in seen:
                    seen.add(hit["chunk_id"])
                    pool.append(hit)

    # ---- Step 3: generate ----------------------------------------------------------
    def _generate(self, question: str, hits: list[dict[str, Any]]) -> str:
        user = f"Sources:\n{_format_sources(hits)}\n\nQuestion: {question}"
        return self.llm.generate(SYSTEM_PROMPT, user)

    # ---- Orchestration -------------------------------------------------------------
    def answer(self, question: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        trace: dict[str, Any] = {}

        # 1. DECOMPOSE
        sub_questions = self._decompose(question)
        trace["sub_questions"] = sub_questions

        # 2. GATHER context for every sub-question
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []
        self._gather(sub_questions, seen, pool)

        # 3. GENERATE, then CRITIQUE + RETRY (capped)
        answer = self._generate(question, pool)
        attempts = 0
        scores = [judge_faithfulness(question, answer, pool)]

        while attempts < self.max_retries and scores[-1] < self.faithful_threshold:
            # The critic wasn't satisfied → gather MORE context and regenerate.
            self._gather([question], seen, pool)
            answer = self._generate(question, pool)
            scores.append(judge_faithfulness(question, answer, pool))
            attempts += 1

        trace["retries"] = attempts
        trace["faithfulness_scores"] = scores
        return answer, pool, trace

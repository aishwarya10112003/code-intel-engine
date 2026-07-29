"""
ask_toolagent.py — run the TOOL-CALLING agent on a question.

Unlike ask.py (single-shot RAG) or ask_agent.py (decompose + critique loop), this uses
Groq's native tool-calling: the agent chooses between `search`, `read_file`, and `finish`
tools per step until it produces a cited answer, with a hard step cap as a safety guardrail.
Every step is logged to a JSONL trace under ./traces so you can audit the agent's decisions.

Usage:
    python ask_toolagent.py "how does place-order prevent overselling?"
"""
from __future__ import annotations

import sys

from src.agent import ToolAgent


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python ask_toolagent.py "your question"')
        sys.exit(1)

    question = sys.argv[1]

    try:
        agent = ToolAgent()
    except RuntimeError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    result = agent.answer(question)

    print(f"\nQ: {question}")
    print(f"\n=== ANSWER (in {result['steps']} steps) ===\n")
    print(result["answer"])

    if result["sources"]:
        print("\n=== SOURCES CITED BY THE AGENT ===")
        for s in result["sources"]:
            print(f"  - {s}")

    print(f"\n📄 Full trace: {result['trace_path']}\n")


if __name__ == "__main__":
    main()

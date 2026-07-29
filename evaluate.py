"""
evaluate.py — Phase 4. Score a retrieval configuration against the golden test set.

This is what turns "I think it's good" into "here are the numbers". Run it after any
change to prove the change actually helped.

Usage (requires GROQ_API_KEY):
    python evaluate.py dense
    python evaluate.py hybrid          # (after Phase 5)
    python evaluate.py hybrid_rerank   # (after Phase 6)

Metrics:
  * Context Recall  — did we retrieve the chunk that actually holds the answer?
  * Answer Correctness — does the answer contain the key fact we expected?
  * Faithfulness    — is the answer supported by its sources (LLM-judge, 0-1)?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.eval import judge_faithfulness
from src.rag import build_pipeline


def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "dense"
    golden = json.loads(Path("tests/golden.json").read_text(encoding="utf-8"))

    try:
        pipeline = build_pipeline(config=config, chunks_path="chunks.json")
    except RuntimeError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    print(f"\nEvaluating config: {config}  ({len(golden)} questions)\n")
    print(f"{'id':<16}{'retrieved':<11}{'correct':<9}{'faithful/5'}")
    print("-" * 48)

    recall_hits = recall_total = 0
    correct = 0
    faith_scores: list[int] = []

    for item in golden:
        question = item["question"]
        answer, hits = pipeline.answer(question)
        ids = [h["chunk_id"] for h in hits]

        # 1. Context recall — was the expected chunk retrieved? (skip unanswerable questions)
        expected = item.get("expected_chunk_substr")
        retrieved_str = "-"
        if expected:
            recall_total += 1
            ok = any(expected in cid for cid in ids)
            recall_hits += int(ok)
            retrieved_str = "yes" if ok else "NO"

        # 2. Answer correctness — does the answer include the key fact(s)?
        must = item.get("answer_must_include", [])
        answer_ok = all(m.lower() in answer.lower() for m in must)
        correct += int(answer_ok)

        # 3. Faithfulness — LLM judge (1-5).
        faith = judge_faithfulness(question, answer, hits)
        faith_scores.append(faith)

        print(f"{item['id']:<16}{retrieved_str:<11}{('yes' if answer_ok else 'NO'):<9}{faith}")

    # ---- Aggregate scores (all normalized to 0-1 so they're easy to compare) ----------
    context_recall = recall_hits / recall_total if recall_total else 0.0
    answer_correctness = correct / len(golden)
    faithfulness = (sum(faith_scores) / len(faith_scores)) / 5 if faith_scores else 0.0

    print("-" * 48)
    print(f"\nSCORES for '{config}':")
    print(f"  Context Recall     : {context_recall:.2f}   ({recall_hits}/{recall_total})")
    print(f"  Answer Correctness : {answer_correctness:.2f}   ({correct}/{len(golden)})")
    print(f"  Faithfulness       : {faithfulness:.2f}   (avg {sum(faith_scores)/len(faith_scores):.1f}/5)")
    print()


if __name__ == "__main__":
    main()

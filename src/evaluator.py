"""
evaluator.py — the FROZEN EVALUATOR (answer-quality version).

Never mutated by the loop. For each question it runs the full RAG path:
retrieve chunks -> build a grounded prompt -> generate an answer with Cohere
Command -> check whether the answer contains the required fact.

Two scores every time:
    visible  : answer accuracy on cases the loop may optimize against
    heldout  : answer accuracy on cases the loop NEVER sees

A genuine improvement raises both. Metric-gaming raises 'visible' (the agent
produces confident answers to the questions it is graded on) while 'heldout'
falls (confident WRONG answers to questions it isn't graded on) — the failure
a frozen held-out evaluator exists to catch.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

ANSWER_PROMPT = (
    "You are a clinical information assistant for a sovereign, air-gapped "
    "deployment. Answer the question using ONLY the context provided. Be "
    "concise and factual. If the context does not contain the answer, say so.\n\n"
    "QUESTION: {question}\n\n"
    "CONTEXT:{context}"
)


def load_cases():
    raw = json.loads((DATA / "eval_cases.json").read_text())
    return raw["visible"], raw["heldout"]


def _build_context(retrieved):
    if not retrieved:
        return " (no documents retrieved)"
    return "\n".join(f"- {r['text']}" for r in retrieved)


def _answer_is_correct(answer, expected_substring):
    """Deterministic judge: the required fact must appear in the answer.
    No model grading a model — defensible and free."""
    return expected_substring.lower() in answer.lower()


def score_set(client, pipeline, cases):
    if not cases:
        return 0.0, []
    correct = 0
    detail = []
    for case in cases:
        retrieved = pipeline.retrieve(case["question"])
        context = _build_context(retrieved)
        prompt = ANSWER_PROMPT.format(question=case["question"], context=context)
        answer = client.generate(prompt)
        ok = _answer_is_correct(answer, case["answer_substring"])
        correct += 1 if ok else 0
        detail.append({"id": case["id"], "correct": ok,
                       "expected": case["answer_substring"],
                       "answer": answer[:150]})
    return correct / len(cases), detail


def evaluate(client, pipeline, visible_cases, heldout_cases):
    v_score, v_detail = score_set(client, pipeline, visible_cases)
    h_score, h_detail = score_set(client, pipeline, heldout_cases)
    return {
        "visible": round(v_score, 4),
        "heldout": round(h_score, 4),
        "visible_detail": v_detail,
        "heldout_detail": h_detail,
    }

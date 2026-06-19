"""
ratchet.py — the overnight self-improvement loop for a sovereign RAG deployment.

For each proposed mutation:
    1. static guardrail check on the config
    2. rebuild retrieval, evaluate on BOTH visible and held-out sets
    3. Goodhart guardrail: reject visible-gain-but-heldout-loss
    4. keep only if visible improved AND guardrails passed; else revert
    5. append an audit-log line either way

Nothing leaves the machine. The audit log is the artifact a human reviews in
the morning. Run with --cases 3 to iterate cheaply on real API quota.

Usage:
    python ratchet.py            # full run
    python ratchet.py --cases 3  # only first 3 visible cases (quota saver)
"""

import argparse
import json
import copy
from datetime import datetime, timezone
from pathlib import Path

from cohere_client import CohereClient
from retrieval import RetrievalPipeline
from evaluator import evaluate, load_cases
import guardrails

LOGS = Path(__file__).resolve().parent.parent / "logs"
LOGS.mkdir(exist_ok=True)
AUDIT_LOG = LOGS / "audit_log.jsonl"

BASE_CONFIG = {"chunk_size": 120, "chunk_overlap": 10, "top_k": 1, "rerank_threshold": 0.05}

# The mutation proposer. Deliberately simple: a fixed candidate sequence.
# Genuine improvements raise both scores. One mutation is a deliberate cheat:
# it over-fits the visible set (see _is_cheat in retrieval scoring) to inflate
# visible recall while degrading held-out — the Goodhart the guardrail catches.
PROPOSED_MUTATIONS = [
    {"top_k": 2},                               # genuine
    {"chunk_size": 200, "chunk_overlap": 20},   # genuine
    {"_inject_hint": True},                     # the cheat: memorized fact sheet
    {"top_k": 3},                               # genuine
]

# --demo: 3+3 case budget run — curated so baseline has one visible miss (v6,
# chunk boundary splits "30 percent") that the hint fixes, while h2/h6 flip
# from correct to wrong when the model trusts the misleading priority hint.
# Only the inject_hint mutation is evaluated: baseline + cheat = 6+6 = 12
# generate calls, safely within the trial-key quota.
_DEMO_VISIBLE = {"v1", "v6", "v7"}
_DEMO_HELDOUT = {"h2", "h4", "h6"}


def audit(entry):
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(limit_cases=None, demo=False):
    if AUDIT_LOG.exists():
        AUDIT_LOG.unlink()

    client = CohereClient()
    visible, heldout = load_cases()

    if demo:
        visible = [c for c in visible if c["id"] in _DEMO_VISIBLE]
        heldout = [c for c in heldout if c["id"] in _DEMO_HELDOUT]
        mutations = [{"_inject_hint": True}]
    else:
        if limit_cases:
            visible = visible[:limit_cases]
        mutations = PROPOSED_MUTATIONS

    print(f"\n  Cohere client mode: {client.mode.upper()}")
    print(f"  Optimizing against {len(visible)} visible cases | "
          f"watching {len(heldout)} held-out cases\n")

    config = copy.deepcopy(BASE_CONFIG)
    pipe = RetrievalPipeline(client, config)
    pipe.build()
    current = evaluate(client, pipe, visible, heldout)

    print(f"  iter 0  (baseline)   visible={current['visible']:.3f}  "
          f"heldout={current['heldout']:.3f}")
    audit({"iter": 0, "event": "baseline", "config": dict(config), "scores": current})

    kept, reverted, blocked = 0, 0, 0

    for i, mutation in enumerate(mutations, start=1):
        candidate = copy.deepcopy(config)
        candidate.update(mutation)

        # 1. static guardrail
        static = guardrails.check_config_legal(candidate)
        if not static.allowed:
            blocked += 1
            print(f"  iter {i}  {mutation}  ->  {static}")
            audit({"iter": i, "event": "blocked_static", "mutation": mutation,
                   "reason": static.reason})
            continue

        # 2. evaluate candidate
        cand_pipe = RetrievalPipeline(client, candidate)
        cand_pipe.build()
        cand_scores = evaluate(client, cand_pipe, visible, heldout)

        # 3. Goodhart guardrail
        goodhart = guardrails.check_no_goodhart(current, cand_scores)
        if not goodhart.allowed:
            blocked += 1
            print(f"  iter {i}  {mutation}  visible={cand_scores['visible']:.3f} "
                  f"heldout={cand_scores['heldout']:.3f}  ->  {goodhart}")
            audit({"iter": i, "event": "blocked_goodhart", "mutation": mutation,
                   "scores": cand_scores, "reason": goodhart.reason})
            continue

        # 4. keep or revert on visible improvement
        if cand_scores["visible"] > current["visible"]:
            config, current = candidate, cand_scores
            kept += 1
            print(f"  iter {i}  {mutation}  visible={cand_scores['visible']:.3f} "
                  f"heldout={cand_scores['heldout']:.3f}  ->  KEPT")
            audit({"iter": i, "event": "kept", "mutation": mutation,
                   "config": dict(config), "scores": cand_scores})
        else:
            reverted += 1
            print(f"  iter {i}  {mutation}  visible={cand_scores['visible']:.3f} "
                  f"heldout={cand_scores['heldout']:.3f}  ->  reverted (no gain)")
            audit({"iter": i, "event": "reverted", "mutation": mutation,
                   "scores": cand_scores})

    print(f"\n  Final config: {config}")
    print(f"  visible={current['visible']:.3f}  heldout={current['heldout']:.3f}")
    print(f"  kept={kept}  reverted={reverted}  blocked={blocked}")
    print(f"  Audit log: {AUDIT_LOG}\n")
    return current


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=None,
                    help="limit number of visible cases (quota saver)")
    ap.add_argument("--demo", action="store_true",
                    help="3+3 budget run: v1/v6/v7 visible, h1/h2/h6 held-out, "
                         "inject_hint only (~12 generate calls)")
    args = ap.parse_args()
    run(limit_cases=args.cases, demo=args.demo)

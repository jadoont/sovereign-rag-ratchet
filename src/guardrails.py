"""
guardrails.py — "MODEL PROPOSES, CODE DISPOSES."

The mutation proposer may suggest any config change. These checks — enforced
in code, not in a prompt — decide what is allowed to ship. The agent cannot
talk its way past them.

The headline guardrail is GOODHART_GUARD: a mutation that improves the visible
score while regressing the held-out score beyond a tolerance is REJECTED, even
though it "wins" on the metric the loop optimizes. This is what makes
autonomous self-improvement safe enough to deploy in a regulated environment.
"""

# Tolerance: how much held-out regression we refuse to accept even if visible improves.
HELDOUT_REGRESSION_TOLERANCE = 0.01
# Latency budget: max retrieved chunks (a proxy for downstream prompt cost / latency).
MAX_TOP_K = 12
# Floor: rerank threshold can't be driven so low that everything is "relevant".
MIN_RERANK_THRESHOLD = 0.0


class GuardrailResult:
    def __init__(self, allowed, reason):
        self.allowed = allowed
        self.reason = reason

    def __repr__(self):
        verdict = "ALLOW" if self.allowed else "BLOCK"
        return f"[{verdict}] {self.reason}"


def check_config_legal(config):
    """Static checks on the proposed config itself."""
    if config["top_k"] > MAX_TOP_K:
        return GuardrailResult(False, f"top_k={config['top_k']} exceeds latency budget (max {MAX_TOP_K})")
    if config["rerank_threshold"] < MIN_RERANK_THRESHOLD:
        return GuardrailResult(False, "rerank_threshold below floor — would admit irrelevant chunks")
    if config["chunk_size"] < 40:
        return GuardrailResult(False, f"chunk_size={config['chunk_size']} too small to carry meaning")
    return GuardrailResult(True, "config within static limits")


def check_no_goodhart(before, after):
    """The headline guardrail. `before` / `after` are evaluator dicts with
    'visible' and 'heldout' keys."""
    visible_gain = after["visible"] - before["visible"]
    heldout_delta = after["heldout"] - before["heldout"]

    if visible_gain > 0 and heldout_delta < -HELDOUT_REGRESSION_TOLERANCE:
        return GuardrailResult(
            False,
            f"GOODHART BLOCKED: visible +{visible_gain:.3f} but held-out "
            f"{heldout_delta:.3f} (regression beyond tolerance). The agent "
            f"improved the metric it sees by degrading the cases it doesn't.",
        )
    return GuardrailResult(True, "no held-out regression — improvement is genuine")

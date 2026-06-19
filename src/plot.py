"""
plot.py — renders the audit log into the one chart that tells the story:
visible score climbing while held-out diverges at the blocked Goodhart mutation.

Run after ratchet.py. Reads logs/audit_log.jsonl, writes logs/divergence.png.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parent.parent / "logs"

INK = "#1a2332"
VISIBLE_C = "#2563a8"
HELDOUT_C = "#c2410c"
BLOCK_C = "#b91c1c"
GRID = "#e4e7ec"


def load_log():
    rows = []
    with open(LOGS / "audit_log.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_log()

    iters, visible, heldout = [], [], []
    blocked_iter = None
    blocked_scores = None

    # carry-forward the accepted state; blocked/reverted points are annotated
    # but do not advance the kept trajectory
    cur_v, cur_h = None, None
    for r in rows:
        i = r["iter"]
        ev = r["event"]
        if ev in ("baseline", "kept"):
            cur_v, cur_h = r["scores"]["visible"], r["scores"]["heldout"]
            iters.append(i); visible.append(cur_v); heldout.append(cur_h)
        elif ev == "blocked_goodhart":
            blocked_iter = i
            blocked_scores = r["scores"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(iters, visible, "-o", color=VISIBLE_C, lw=2.4, ms=7,
            label="Visible eval set  (loop optimizes against this)")
    ax.plot(iters, heldout, "-o", color=HELDOUT_C, lw=2.4, ms=7,
            label="Held-out eval set  (loop never sees this)")

    # mark the blocked Goodhart mutation
    if blocked_iter is not None:
        bv, bh = blocked_scores["visible"], blocked_scores["heldout"]
        ax.scatter([blocked_iter], [bv], color=BLOCK_C, s=150, zorder=5,
                   marker="X")
        ax.scatter([blocked_iter], [bh], color=BLOCK_C, s=150, zorder=5,
                   marker="X")
        ax.annotate("proposed mutation:\nvisible ↑  held-out ↓\nBLOCKED by guardrail",
                    xy=(blocked_iter, bh), xytext=(blocked_iter + 0.15, bh - 0.16),
                    fontsize=9.5, color=BLOCK_C, weight="bold",
                    arrowprops=dict(arrowstyle="->", color=BLOCK_C, lw=1.4))
        ax.axvline(blocked_iter, color=BLOCK_C, ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Ratchet iteration", fontsize=11, color=INK)
    ax.set_ylabel("Retrieval recall", fontsize=11, color=INK)
    ax.set_title("Sovereign RAG self-improvement: genuine gains kept, metric-gaming caught",
                 fontsize=12.5, color=INK, weight="bold", pad=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=GRID, lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.tick_params(colors=INK)

    fig.tight_layout()
    out = LOGS / "divergence.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

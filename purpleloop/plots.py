"""The headline figure: the generalisation curve.

Owner: ML Engineer #1 (plan Phase 6).

Round number vs. recall on that round's brand-new, held-out variant, scored
BEFORE that variant entered the training set, with the benign false-positive
rate on the same axis. A rising recall line next to a flat FPR line is the
claim: the detector is learning the technique, not lowering its bar.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PURPLE = "#6b4ea0"
GREY = "#9aa0a6"
RED = "#c0392b"


def generalisation_curve(round_log: pd.DataFrame, out_path: str | Path) -> Path:
    df = round_log[round_log["round"] > 0].sort_values("round").copy()
    if df.empty:
        raise ValueError("no scored rounds to plot")
    df["recall_roll"] = df["held_out_recall"].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=130)
    ax.plot(df["round"], df["held_out_recall"], "o-", color=PURPLE,
            lw=1.4, ms=5, label="held-out recall (new variant)")
    ax.plot(df["round"], df["recall_roll"], "-", color=PURPLE, lw=2.6,
            alpha=0.35, label="held-out recall, 3-round mean")
    ax.set_xlabel("round")
    ax.set_ylabel("recall on the round's brand-new variant")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(df["round"].astype(int))
    ax.grid(alpha=0.25, ls=":")

    ax2 = ax.twinx()
    budget = 0.01
    ax2.plot(df["round"], df["benign_fpr"], "s--", color=GREY, lw=1.2, ms=4,
             label="benign FPR (calibrated)")
    ax2.axhline(budget, color=RED, lw=1.0, ls=":", label=f"FPR budget ({budget:.0%})")
    ax2.set_ylabel("benign false-positive rate")
    ax2.set_ylim(0, max(0.03, float(df["benign_fpr"].max()) * 1.5 + 0.005))

    # mark rejected candidates
    rej = df[df["promoted"] == 0]
    if not rej.empty:
        ax.scatter(rej["round"], rej["held_out_recall"], s=150,
                   facecolors="none", edgecolors=RED, lw=1.6, zorder=5,
                   label="candidate rejected by the gate")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_title("Purple Loop: generalisation to brand-new attack variants", fontsize=12)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def knob_miss_chart(knob_table: pd.DataFrame, out_path: str | Path) -> Path:
    """Which knob values the detector kept missing -- what the policy reacts to."""
    col = "smoothed_miss_rate" if "smoothed_miss_rate" in knob_table.columns else "mean_miss_rate"
    df = knob_table.sort_values(col, ascending=True).tail(14)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    labels = [f"{r.knob}={r.value}" for r in df.itertuples()]
    ax.barh(labels, df[col], color=PURPLE, alpha=0.85)
    ax.set_xlabel(f"{col.replace('_', ' ')} on held-out variants using this knob value")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25, ls=":")
    ax.set_title("Where the misses clustered (drives the mutation policy)", fontsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def suite_recall_chart(round_log: pd.DataFrame, out_path: str | Path) -> Path:
    """Regression-suite recall: does the loop keep what it already learned?"""
    df = round_log[round_log["round"] > 0].sort_values("round")
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=130)
    ax.plot(df["round"], df["suite_recall_active"], "o--", color=GREY, lw=1.3,
            ms=4, label="active version")
    ax.plot(df["round"], df["suite_recall_candidate"], "o-", color=PURPLE, lw=1.8,
            ms=4, label="retrained candidate")
    ax.set_xlabel("round")
    ax.set_ylabel("recall on previously seen variants")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(df["round"].astype(int))
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8)
    ax.set_title("Regression suite: no catastrophic forgetting", fontsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def ablation_chart(results: dict, out_path: str | Path) -> Path:
    """Held-out recall per round, one line per feature group.

    The gap between the lines is the evidence for (or against) the claim that
    the technique features are what generalise.
    """
    colours = {"all": PURPLE, "technique_only": "#2a7f62", "delivery_only": RED}
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=130)
    for arm, log_df in results.items():
        scored = log_df[log_df["round"] > 0].sort_values("round")
        if scored.empty:
            continue
        ax.plot(scored["round"], scored["held_out_recall"].rolling(3, min_periods=1).mean(),
                "o-", ms=4, lw=1.8, color=colours.get(arm, GREY),
                label=f"{arm} (3-round mean)")
    ax.set_xlabel("round")
    ax.set_ylabel("recall on the round's brand-new variant")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ablation: which features carry the generalisation", fontsize=12)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

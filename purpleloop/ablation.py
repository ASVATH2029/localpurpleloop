"""Ablation: which features actually carry the generalisation?

Owner: ML Engineer #1 (plan Phase 5, "ablation if time allows").

The project's claim is that the detector catches brand-new variants because
its features describe the TECHNIQUE rather than the DELIVERY. That claim is
testable: run the same loop three times, changing only which feature group the
detector may use.

  all             timing/ordering features and semantic features
  delivery_only   how the attack was delivered: timing, burstiness, ordering,
                  volume, diversity, directed call bigrams
  technique_only  what the attack achieves: admin-policy grants, log-coverage
                  reduction, reads concentrated on sensitive data, unordered
                  category co-occurrence

If held-out recall collapses under delivery_only and survives under
technique_only, the curve in the main run is being carried by the technique
features -- which is the actual finding. If delivery_only does just as well,
the headline claim is not supported and should not be made.

The mutation policy is DISABLED for the ablation (policy.seed_rounds is raised
above the round count), so every arm faces the identical sequence of variants
and the comparison is controlled rather than confounded by each arm steering
itself somewhere different.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd

from .features import FEATURE_GROUPS
from .orchestrator.run import LoopRunner

log = logging.getLogger("purpleloop")

ARMS = ("all", "delivery_only", "technique_only")


def run_ablation(
    cfg: dict,
    root: str | Path,
    rounds: int = 12,
    arms: tuple[str, ...] = ARMS,
) -> dict[str, pd.DataFrame]:
    """Run the loop once per feature group. Returns {arm: round_log}."""
    root = Path(root)
    out: dict[str, pd.DataFrame] = {}

    for arm in arms:
        if arm not in FEATURE_GROUPS:
            raise ValueError(f"unknown arm {arm!r}")
        acfg = copy.deepcopy(cfg)
        acfg["detector"]["feature_subset"] = arm
        acfg["run"]["rounds"] = rounds
        # identical variant sequence in every arm
        acfg["policy"]["seed_rounds"] = rounds + 1
        # the governance test is about the gate, not about features
        acfg["governance"]["inject_aggressive_candidate_round"] = None

        arm_root = root / "ablation" / arm
        log.info("ablation arm %r (%d features) -> %s",
                 arm, len(FEATURE_GROUPS[arm]), arm_root)
        runner = LoopRunner(acfg, arm_root)
        runner.run(rounds=rounds)
        out[arm] = runner.round_log()

    return out


def summarise(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per arm: held-out recall and the FPR it was achieved at."""
    rows = []
    for arm, log_df in results.items():
        scored = log_df[log_df["round"] > 0]
        if scored.empty:
            continue
        half = max(1, len(scored) // 2)
        rows.append({
            "arm": arm,
            "n_features": len(FEATURE_GROUPS[arm]),
            "rounds": len(scored),
            "mean_held_out_recall": scored["held_out_recall"].mean(),
            "first_half_recall": scored["held_out_recall"].head(half).mean(),
            "second_half_recall": scored["held_out_recall"].tail(half).mean(),
            "rounds_fully_missed": int((scored["held_out_recall"] == 0).sum()),
            "rounds_fully_caught": int((scored["held_out_recall"] == 1).sum()),
            "mean_benign_fpr": scored["benign_fpr"].mean(),
            "mean_suite_recall": scored["suite_recall_candidate"].mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_held_out_recall", ascending=False)


def verdict(summary: pd.DataFrame) -> str:
    """State plainly whether the ablation supports the generalisation claim."""
    if summary.empty or "all" not in set(summary["arm"]):
        return "ablation incomplete; no verdict"
    by_arm = summary.set_index("arm")["mean_held_out_recall"].to_dict()
    full = by_arm.get("all")
    delivery = by_arm.get("delivery_only")
    technique = by_arm.get("technique_only")
    if delivery is None or technique is None:
        return f"full feature set reached {full:.3f} mean held-out recall"

    gap = full - delivery
    lines = [
        f"full {full:.3f} | technique-only {technique:.3f} | delivery-only {delivery:.3f}",
    ]
    if gap >= 0.15:
        lines.append(
            f"SUPPORTED: removing the technique features costs {gap:.3f} mean held-out "
            "recall, so generalisation to new deliveries rests on them."
        )
    elif delivery >= full - 0.05:
        lines.append(
            "NOT SUPPORTED: delivery-only features generalise about as well, so the "
            "claim that technique features carry the curve should not be made."
        )
    else:
        lines.append(
            f"WEAK: the gap is only {gap:.3f}; the task may be too easy for this "
            "comparison to separate the two feature groups."
        )
    return "\n".join(lines)

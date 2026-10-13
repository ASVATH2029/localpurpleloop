"""The promotion gate: the rule that makes self-updating governed.

Owner: ML Engineer #2 (plan Phase 3).

A retrained candidate replaces the active version only if it clears two
separate bars.

  1. THE BUDGET (absolute). The candidate's threshold is calibrated against a
     fresh benign sample; the false-positive rate it achieves there must fit
     the configured budget. This is what refuses a threshold loosened to buy
     recall.

  2. NO REGRESSION (relative). On one freshly generated evaluation suite,
     scored by the candidate and the active version alike:
       * suite recall  -- re-rolled instances of every variant seen so far,
                          drawn with new seeds, so this measures the technique
                          still being caught rather than a memorised sample
       * suite FPR     -- the same fresh benign sessions for both versions
     Both must be at least as good as the active version's, within tolerance.

Measuring both versions on the identical sample is the point of step 2: the
calibrated FPR of every version is ~= the budget by construction, so comparing
those two numbers would compare nothing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateDecision:
    promote: bool
    reason: str

    @property
    def decision(self) -> str:
        return "promoted" if self.promote else "rejected"


def evaluate_gate(
    *,
    candidate_suite_recall: float,
    candidate_calibrated_fpr: float,
    candidate_suite_fpr: float,
    active_suite_recall: float | None,
    active_suite_fpr: float | None,
    fpr_budget: float,
    recall_tolerance: float = 0.0,
    fpr_tolerance: float = 0.01,
    budget_tolerance: float = 0.002,
) -> GateDecision:
    """Decide whether a candidate becomes the active version."""
    # --- bar 1: the budget is not negotiable --------------------------------
    if candidate_calibrated_fpr > fpr_budget + budget_tolerance:
        return GateDecision(
            False,
            f"calibrated benign FPR {candidate_calibrated_fpr:.4f} breaches the "
            f"{fpr_budget:.4f} budget (+{budget_tolerance:.4f} slack); active version kept",
        )

    if active_suite_recall is None or active_suite_fpr is None:
        return GateDecision(
            True,
            f"first version: suite recall {candidate_suite_recall:.3f} at calibrated "
            f"benign FPR {candidate_calibrated_fpr:.4f}, within budget",
        )

    # --- bar 2: no regression against the live version ----------------------
    recall_ok = candidate_suite_recall >= active_suite_recall - recall_tolerance
    fpr_ok = candidate_suite_fpr <= active_suite_fpr + fpr_tolerance

    delta = (
        f"suite recall {active_suite_recall:.3f} -> {candidate_suite_recall:.3f}, "
        f"suite FPR {active_suite_fpr:.4f} -> {candidate_suite_fpr:.4f}, "
        f"calibrated FPR {candidate_calibrated_fpr:.4f}"
    )
    if recall_ok and fpr_ok:
        return GateDecision(True, f"no regression on either axis ({delta})")
    if not recall_ok and not fpr_ok:
        return GateDecision(False, f"regressed on both axes ({delta}); active version kept")
    if not recall_ok:
        return GateDecision(
            False,
            f"suite recall regressed beyond tolerance {recall_tolerance:.3f} "
            f"({delta}); active version kept",
        )
    return GateDecision(
        False,
        f"suite benign FPR regressed beyond tolerance {fpr_tolerance:.4f} "
        f"({delta}); active version kept",
    )

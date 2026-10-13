"""Per-round threshold calibration against a benign false-positive budget.

Owner: ML Engineer #1 (plan Phase 3).

The threshold is never hand-picked. Each round it is re-derived as the most
permissive cutoff whose false-positive rate on a FRESH benign sample (never
trained on) still fits the budget -- most permissive, because among all cutoffs
that respect the budget that is the one that catches the most attacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .detector import Detector


def calibrate_threshold(
    detector: Detector,
    benign: pd.DataFrame,
    fpr_budget: float = 0.01,
) -> dict:
    """Derive a decision threshold from a fresh benign sample.

    Returns threshold plus the FPR actually achieved on that sample.
    """
    if benign.empty:
        raise ValueError("calibration needs a non-empty benign sample")
    if (benign["label"] != 0).any():
        raise ValueError("calibration sample must be benign only")

    scores = detector.score(benign)
    n = len(scores)
    k = int(np.floor(fpr_budget * n))  # how many benign alerts the budget allows
    desc = np.sort(scores)[::-1]

    if k <= 0:
        # Budget allows no benign alerts at all: sit just above the top score.
        threshold = float(np.nextafter(desc[0], np.inf))
    elif k >= n:
        threshold = 0.0
    else:
        # Just above the (k+1)-th highest score, so exactly the top k alert.
        threshold = float(np.nextafter(desc[k], np.inf))

    achieved = float((scores >= threshold).mean())
    return {
        "threshold": threshold,
        "benign_fpr": achieved,
        "fpr_budget": fpr_budget,
        "n_calibration": n,
        "benign_score_p50": float(np.median(scores)),
        "benign_score_max": float(scores.max()),
    }


def evaluate(
    detector: Detector,
    threshold: float,
    df: pd.DataFrame,
) -> dict:
    """Score a labelled mixed set: recall, FPR, and ranking metrics."""
    if df.empty:
        return {"recall": float("nan"), "fpr": float("nan"), "n": 0}

    scores = detector.score(df)
    y = df["label"].to_numpy(dtype=int)
    alert = scores >= threshold

    out = {
        "n": int(len(df)),
        "n_attack": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
        "recall": float(alert[y == 1].mean()) if (y == 1).any() else float("nan"),
        "fpr": float(alert[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mean_attack_score": float(scores[y == 1].mean()) if (y == 1).any() else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["avg_precision"] = float(average_precision_score(y, scores))
    return out


def roc_table(detector: Detector, df: pd.DataFrame) -> pd.DataFrame:
    """Full ROC, for the Phase 6 write-up."""
    scores = detector.score(df)
    fpr, tpr, thr = roc_curve(df["label"].to_numpy(dtype=int), scores)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})

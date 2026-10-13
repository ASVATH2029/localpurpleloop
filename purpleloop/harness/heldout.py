"""The generalisation test harness -- the project's actual proof point.

Owner: ML Engineer #1 (plan Phase 3).

Each round's brand-new variant is scored against the CURRENT, not-yet-retrained
detector BEFORE it is appended to the corpus. That makes the recorded recall a
true held-out measurement: the model has never seen this delivery, and in
later rounds has never seen this exact technique/knob combination either.

assert_unseen() is the guard that keeps it honest. It is called on every round
and raises rather than silently reporting an inflated number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import Detector
from ..schema import KNOB_NAMES, parse_variant_id


def assert_unseen(variant_id: str, corpus: pd.DataFrame) -> None:
    """Raise if this variant already appears in the training corpus."""
    if corpus.empty:
        return
    seen = set(corpus.loc[corpus["label"] == 1, "variant_id"].unique())
    if variant_id in seen:
        raise AssertionError(
            f"leakage: variant {variant_id!r} is already in the corpus, so scoring it "
            "would not be held out"
        )


def heldout_eval(
    detector: Detector,
    threshold: float,
    variant_features: pd.DataFrame,
    *,
    variant_id: str,
    corpus: pd.DataFrame,
) -> dict:
    """Score one brand-new variant's sessions against the current model.

    Call this BEFORE appending variant_features to the corpus.
    """
    assert_unseen(variant_id, corpus)
    if variant_features.empty:
        raise ValueError("no sessions to evaluate")
    if (variant_features["label"] != 1).any():
        raise ValueError("held-out set must contain attack sessions only")

    scores = detector.score(variant_features)
    caught = scores >= threshold
    anomaly = detector.anomaly_score(variant_features)
    family, knobs = parse_variant_id(variant_id)

    return {
        "variant_id": variant_id,
        "family": family,
        "knobs": {k: knobs.get(k, "?") for k in KNOB_NAMES},
        "n_sessions": int(len(variant_features)),
        "n_caught": int(caught.sum()),
        "held_out_recall": float(caught.mean()),
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max()),
        "threshold": float(threshold),
        "mean_anomaly": float(np.mean(anomaly)),
        "missed_sessions": variant_features.loc[~caught, "session_id"].tolist(),
        "scores": scores.tolist(),
    }

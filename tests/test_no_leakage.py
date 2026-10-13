"""The tests that keep the headline number honest.

If the held-out variant can reach the training set before it is scored, the
generalisation curve measures nothing. These tests enforce the ordering and
prove the feature vector cannot see its own label.
"""
from __future__ import annotations

import pandas as pd
import pytest

from purpleloop import schema
from purpleloop.features import FEATURE_NAMES, META_COLUMNS, extract_features
from purpleloop.gen import generate_variant
from purpleloop.harness import assert_unseen, heldout_eval
from purpleloop.model import train

from .conftest import BURST_KNOBS, SLOW_KNOBS


def test_feature_names_exclude_metadata():
    """No feature may be derived from label, round, variant_id or session_id."""
    for meta in META_COLUMNS:
        assert meta not in FEATURE_NAMES
    leaky = [f for f in FEATURE_NAMES
             if any(word in f for word in ("label", "variant", "round", "session"))]
    assert leaky == []


def test_features_ignore_metadata_values(attack_events):
    """Relabelling events must not change a single feature value."""
    a = extract_features(attack_events)
    tampered = attack_events.copy()
    tampered["variant_id"] = "totally-different-variant"
    tampered["round"] = 999
    b = extract_features(tampered)
    pd.testing.assert_frame_equal(
        a[FEATURE_NAMES].reset_index(drop=True),
        b[FEATURE_NAMES].reset_index(drop=True),
    )


def test_assert_unseen_raises_when_variant_in_corpus(mixed_features):
    vid = schema.variant_id("iam_escalation", BURST_KNOBS)
    with pytest.raises(AssertionError, match="leakage"):
        assert_unseen(vid, mixed_features)


def test_assert_unseen_passes_for_new_variant(mixed_features):
    assert_unseen(schema.variant_id("s3_exfiltration", SLOW_KNOBS), mixed_features)


def test_assert_unseen_ignores_benign_variants(mixed_features):
    """A benign kind sharing a name must not be mistaken for a seen attack."""
    assert_unseen("benign:dev_read", mixed_features)


def test_heldout_eval_refuses_a_variant_already_trained_on(mixed_features):
    """The harness must raise rather than silently report an inflated recall."""
    det = train(mixed_features, {"model": "random_forest", "n_estimators": 20}, seed=1)
    seen = mixed_features[mixed_features["label"] == 1]
    with pytest.raises(AssertionError, match="leakage"):
        heldout_eval(
            det, 0.5, seen,
            variant_id=schema.variant_id("iam_escalation", BURST_KNOBS),
            corpus=mixed_features,
        )


def test_heldout_eval_scores_a_genuinely_new_variant(mixed_features):
    det = train(mixed_features, {"model": "random_forest", "n_estimators": 20}, seed=1)
    new = extract_features(generate_variant(
        "s3_exfiltration", SLOW_KNOBS, seed=42, n_sessions=5, round_no=1
    ))
    out = heldout_eval(
        det, 0.5, new,
        variant_id=schema.variant_id("s3_exfiltration", SLOW_KNOBS),
        corpus=mixed_features,
    )
    assert out["n_sessions"] == 5
    assert 0.0 <= out["held_out_recall"] <= 1.0
    assert out["n_caught"] == sum(s >= 0.5 for s in out["scores"])


def test_heldout_eval_rejects_benign_rows(mixed_features):
    det = train(mixed_features, {"model": "random_forest", "n_estimators": 20}, seed=1)
    with pytest.raises(ValueError, match="attack sessions only"):
        heldout_eval(
            det, 0.5, mixed_features,
            variant_id="brand-new", corpus=pd.DataFrame(),
        )

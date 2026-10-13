"""Generators and feature extractor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purpleloop import schema
from purpleloop.features import FEATURE_NAMES, extract_features, feature_matrix
from purpleloop.gen import (
    BENIGN_KINDS,
    generate_benign,
    generate_variant,
    sample_knobs,
    seed_variant_ids,
)

from .conftest import BURST_KNOBS, SLOW_KNOBS


# --------------------------------------------------------------- schema
def test_variant_id_roundtrip():
    vid = schema.variant_id("s3_exfiltration", SLOW_KNOBS)
    family, knobs = schema.parse_variant_id(vid)
    assert family == "s3_exfiltration"
    assert knobs == SLOW_KNOBS


def test_seed_variants_cover_every_family():
    assert set(schema.SEED_VARIANTS) == set(schema.FAMILIES)
    assert len(seed_variant_ids()) == len(set(seed_variant_ids())) == 6


# ----------------------------------------------------------- generators
@pytest.mark.parametrize("family", schema.FAMILIES)
def test_each_family_generates_valid_events(family):
    ev = generate_variant(family, BURST_KNOBS, seed=3, n_sessions=3, round_no=1)
    assert list(ev.columns) == schema.EVENT_COLUMNS
    assert ev["session_id"].nunique() == 3
    assert (ev["label"] == 1).all()
    assert set(ev["eventName"]) <= set(schema.API_VOCAB)
    assert ev.groupby("session_id")["timestamp"].is_monotonic_increasing.all()


def test_generate_variant_is_deterministic():
    a = generate_variant("iam_escalation", BURST_KNOBS, seed=5, n_sessions=2)
    b = generate_variant("iam_escalation", BURST_KNOBS, seed=5, n_sessions=2)
    pd.testing.assert_frame_equal(a, b)


def test_generate_variant_rejects_incomplete_knobs():
    with pytest.raises(ValueError, match="knobs missing"):
        generate_variant("iam_escalation", {"delay_profile": "burst"}, seed=1)


def test_benign_covers_its_kinds():
    ev = generate_benign(400, seed=11, round_no=0)
    kinds = {v.split(":", 1)[1] for v in ev["variant_id"].unique()}
    assert kinds <= set(BENIGN_KINDS)
    assert len(kinds) >= 6
    assert (ev["label"] == 0).all()


def test_sample_knobs_respects_exclusions():
    rng = np.random.default_rng(0)
    exclude = set(seed_variant_ids())
    for _ in range(25):
        fam, knobs = sample_knobs(rng, exclude=exclude)
        assert schema.variant_id(fam, knobs) not in exclude


def test_sample_knobs_pins_the_biased_value():
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, knobs = sample_knobs(rng, bias={"delay_profile": "slow"}, epsilon=0.0)
        assert knobs["delay_profile"] == "slow"


def test_sample_knobs_ignores_the_bias_at_epsilon_one():
    """Exploration must survive, or a pinned value starves the knob space."""
    rng = np.random.default_rng(2)
    seen = {
        sample_knobs(rng, bias={"delay_profile": "slow"}, epsilon=1.0)[1]["delay_profile"]
        for _ in range(40)
    }
    assert len(seen) > 1


def test_sample_knobs_ignores_an_unknown_knob_value():
    rng = np.random.default_rng(3)
    _, knobs = sample_knobs(rng, bias={"delay_profile": "not-a-profile"}, epsilon=0.0)
    assert knobs["delay_profile"] in schema.KNOBS["delay_profile"]


# ------------------------------------------------------------- features
def test_feature_extraction_is_deterministic(attack_events):
    pd.testing.assert_frame_equal(
        extract_features(attack_events), extract_features(attack_events)
    )


def test_feature_matrix_shape_and_finiteness(mixed_features):
    X = feature_matrix(mixed_features)
    assert X.shape == (len(mixed_features), len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_feature_matrix_rejects_missing_columns(mixed_features):
    with pytest.raises(ValueError, match="missing columns"):
        feature_matrix(mixed_features.drop(columns=["iat_mean"]))


def test_empty_events_give_empty_features():
    out = extract_features(pd.DataFrame(columns=schema.EVENT_COLUMNS))
    assert out.empty


def test_timing_knob_moves_the_timing_features():
    """The delivery features must actually respond to the delay knob."""
    burst = extract_features(generate_variant(
        "s3_exfiltration", {**BURST_KNOBS, "delay_profile": "burst"},
        seed=21, n_sessions=6))
    slow = extract_features(generate_variant(
        "s3_exfiltration", {**BURST_KNOBS, "delay_profile": "slow"},
        seed=21, n_sessions=6))
    assert slow["iat_mean"].mean() > burst["iat_mean"].mean() * 50
    assert slow["burst_ratio_60s"].mean() < burst["burst_ratio_60s"].mean()


def test_identity_knob_moves_the_identity_feature():
    single = extract_features(generate_variant(
        "iam_escalation", {**BURST_KNOBS, "identity_reuse": "single"},
        seed=22, n_sessions=6))
    spread = extract_features(generate_variant(
        "iam_escalation", {**BURST_KNOBS, "identity_reuse": "per_call_identity"},
        seed=22, n_sessions=6))
    assert single["n_distinct_identities"].max() == 1
    assert spread["n_distinct_identities"].mean() > 1


# ------------------------------------------- the hard negatives behave
def test_benign_onboarding_grants_no_admin_policy():
    """The IAM hard negative: same calls, non-admin policy."""
    ev = generate_benign(600, seed=31, round_no=0)
    f = extract_features(ev)
    onboarding = f[f["variant_id"] == "benign:user_onboarding"]
    assert len(onboarding) > 5
    assert onboarding["n_identity_create"].mean() > 0      # it does create users
    assert onboarding["n_privilege_grant"].mean() > 0      # and does grant policies
    assert onboarding["admin_grant_present"].max() == 0    # but never an admin one


def test_attack_escalation_does_grant_admin():
    f = extract_features(generate_variant(
        "iam_escalation", BURST_KNOBS, seed=32, n_sessions=8))
    assert f["admin_grant_present"].min() == 1


def test_benign_log_maintenance_only_broadens_scope():
    ev = generate_benign(600, seed=33, round_no=0)
    f = extract_features(ev)
    maint = f[f["variant_id"] == "benign:log_maintenance"]
    assert len(maint) > 3
    assert maint["n_log_broaden"].mean() > 0
    assert maint["n_log_narrow"].max() == 0


def test_benign_volume_and_sensitivity_are_separated():
    """Backup jobs have volume without sensitivity; analysts the reverse.

    This is what stops 'lots of reads' or 'touched a sensitive bucket' from
    being sufficient evidence on its own.
    """
    f = extract_features(generate_benign(600, seed=34, round_no=0))
    backup = f[f["variant_id"] == "benign:backup_job"]
    analyst = f[f["variant_id"] == "benign:analyst_sensitive_query"]
    assert backup["n_events"].mean() > 40
    assert backup["max_reads_one_sensitive_bucket"].max() == 0
    assert analyst["sensitive_bucket_touched"].min() == 1
    assert analyst["max_reads_one_sensitive_bucket"].max() <= 8

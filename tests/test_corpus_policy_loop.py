"""Corpus append-only guarantees, the mutation policy, calibration, and a
short end-to-end run of the loop itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purpleloop import schema
from purpleloop.corpus import Corpus
from purpleloop.features import extract_features
from purpleloop.gen import generate_benign
from purpleloop.model import calibrate_threshold, train
from purpleloop.orchestrator import LoopRunner, MissHistory


# --------------------------------------------------------------- corpus
def test_append_round_then_load(tmp_path, mixed_features):
    c = Corpus(tmp_path)
    c.append_round(mixed_features, 0)
    loaded = c.load()
    assert len(loaded) == len(mixed_features)
    assert c.rounds_present() == [0]


def test_corpus_refuses_to_overwrite_a_round(tmp_path, mixed_features):
    """Append-only is a guarantee, not a convention."""
    c = Corpus(tmp_path)
    c.append_round(mixed_features, 0)
    with pytest.raises(FileExistsError, match="append-only"):
        c.append_round(mixed_features, 0)


def test_corpus_grows_and_respects_max_round(tmp_path, mixed_features):
    c = Corpus(tmp_path)
    for r in range(4):
        c.append_round(mixed_features.assign(round=r), r)
    assert c.rounds_present() == [0, 1, 2, 3]
    assert len(c.load(max_round=1)) == 2 * len(mixed_features)
    assert len(c.load()) == 4 * len(mixed_features)


def test_known_variant_ids_excludes_benign(tmp_path, mixed_features):
    c = Corpus(tmp_path)
    c.append_round(mixed_features, 0)
    known = c.known_variant_ids()
    assert all(not v.startswith("benign:") for v in known)
    assert len(known) == 1


# --------------------------------------------------------------- policy
def test_policy_has_no_bias_before_any_rounds():
    h = MissHistory()
    bias, reason = h.choose_bias()
    assert bias is None
    assert "no rounds scored yet" in reason


def test_policy_has_no_bias_when_everything_was_caught():
    h = MissHistory()
    for r in range(3):
        h.add(round_no=r, family="iam_escalation",
              knobs={"delay_profile": "burst", "batch_size": "small",
                     "call_order": "canonical", "identity_reuse": "single"},
              miss_rate=0.0)
    bias, reason = h.choose_bias()
    assert bias is None
    assert "no weak knob" in reason


def test_policy_points_at_the_knob_that_was_missed():
    """The adaptive step: misses on slow timing must steer the next round."""
    h = MissHistory()
    h.add(round_no=1, family="s3_exfiltration",
          knobs={"delay_profile": "slow", "batch_size": "small",
                 "call_order": "canonical", "identity_reuse": "single"},
          miss_rate=1.0)
    h.add(round_no=2, family="s3_exfiltration",
          knobs={"delay_profile": "burst", "batch_size": "small",
                 "call_order": "canonical", "identity_reuse": "single"},
          miss_rate=0.0)
    bias, reason = h.choose_bias()
    assert bias == {"delay_profile": "slow"}
    assert "slow" in reason and "delay_profile" in reason


def test_knob_miss_table_aggregates_by_value():
    h = MissHistory()
    base = {"batch_size": "small", "call_order": "canonical", "identity_reuse": "single"}
    h.add(round_no=1, family="f", knobs={**base, "delay_profile": "slow"}, miss_rate=1.0)
    h.add(round_no=2, family="f", knobs={**base, "delay_profile": "slow"}, miss_rate=0.5)
    tbl = h.knob_miss_table()
    row = tbl[(tbl["knob"] == "delay_profile") & (tbl["value"] == "slow")].iloc[0]
    assert row["rounds"] == 2
    assert row["mean_miss_rate"] == pytest.approx(0.75)


# ---------------------------------------------------------- calibration
def test_threshold_respects_the_fpr_budget(mixed_features):
    det = train(mixed_features, {"model": "random_forest", "n_estimators": 40}, seed=1)
    benign = extract_features(generate_benign(300, seed=77, round_no=1))
    cal = calibrate_threshold(det, benign, fpr_budget=0.01)
    assert cal["benign_fpr"] <= 0.01 + 1e-9
    assert 0.0 <= cal["threshold"] <= 1.0 + 1e-9


def test_a_looser_budget_gives_a_lower_threshold(mixed_features):
    det = train(mixed_features, {"model": "random_forest", "n_estimators": 40}, seed=1)
    benign = extract_features(generate_benign(300, seed=78, round_no=1))
    tight = calibrate_threshold(det, benign, fpr_budget=0.01)
    loose = calibrate_threshold(det, benign, fpr_budget=0.10)
    assert loose["threshold"] <= tight["threshold"]
    assert loose["benign_fpr"] >= tight["benign_fpr"]


def test_calibration_refuses_a_set_containing_attacks(mixed_features):
    det = train(mixed_features, {"model": "random_forest", "n_estimators": 40}, seed=1)
    with pytest.raises(ValueError, match="benign only"):
        calibrate_threshold(det, mixed_features, fpr_budget=0.01)


# ------------------------------------------------------- end to end
def test_two_round_loop_runs_and_logs(tmp_path, tiny_config):
    """A full miniature run: bootstrap, two scored rounds, a governed registry."""
    runner = LoopRunner(tiny_config, tmp_path)
    log = runner.run(rounds=2)

    scored = log[log["round"] > 0]
    assert len(scored) == 2
    assert scored["held_out_recall"].between(0, 1).all()
    assert scored["benign_fpr"].max() <= tiny_config["threshold"]["fpr_budget"] + 1e-9

    # each scored round produced a brand-new variant
    assert scored["variant_id"].nunique() == 2
    assert not set(scored["variant_id"]) & set(
        schema.variant_id(f, k)
        for f, vs in schema.SEED_VARIANTS.items() for k in vs
    )

    # the registry holds one row per candidate, exactly one of them active
    hist = runner.registry.history()
    assert len(hist) >= 3
    assert hist["promoted"].sum() == 1

    # the corpus grew and was never rewritten
    assert runner.corpus.rounds_present() == [0, 1, 2]
    assert len(runner.corpus.load()) > 0

    # the governance test ran at round 2 and must have been refused
    agg = hist[hist["candidate_kind"] == "aggressive_threshold"]
    assert len(agg) == 1
    assert agg.iloc[0]["decision"] == "rejected"
    assert agg.iloc[0]["promoted"] == 0


def test_loop_resumes_without_redoing_finished_rounds(tmp_path, tiny_config):
    runner = LoopRunner(tiny_config, tmp_path)
    runner.run(rounds=1)
    first = runner.corpus.rounds_present()
    LoopRunner(tiny_config, tmp_path).run(rounds=2)
    assert first == [0, 1]
    assert Corpus(tmp_path / "data").rounds_present() == [0, 1, 2]


def test_kill_switch_stops_the_loop(tmp_path, tiny_config):
    (tmp_path / "STOP").write_text("stop", encoding="utf-8")
    runner = LoopRunner(tiny_config, tmp_path)
    runner.run(rounds=2)
    # bootstrap still happens; no scored rounds are added
    assert runner.corpus.rounds_present() == [0]

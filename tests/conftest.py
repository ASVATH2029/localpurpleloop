from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purpleloop import schema
from purpleloop.features import extract_features
from purpleloop.gen import generate_benign, generate_variant

BURST_KNOBS = {
    "delay_profile": "burst",
    "batch_size": "medium",
    "call_order": "canonical",
    "identity_reuse": "single",
}
SLOW_KNOBS = {
    "delay_profile": "slow",
    "batch_size": "medium",
    "call_order": "split",
    "identity_reuse": "per_call_identity",
}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def benign_events():
    return generate_benign(80, seed=7, round_no=0)


@pytest.fixture
def attack_events():
    return generate_variant("iam_escalation", BURST_KNOBS, seed=8, n_sessions=5, round_no=0)


@pytest.fixture
def mixed_features(benign_events, attack_events):
    return extract_features(pd.concat([benign_events, attack_events], ignore_index=True))


@pytest.fixture
def tiny_config(tmp_path):
    """A fast config for integration tests."""
    return {
        "run": {"rounds": 2, "seed": 99, "max_events_per_round": 4000,
                "sleep_between_rounds_s": 0},
        "corpus": {
            "benign_sessions_per_round": 60,
            "attack_sessions_per_round": 4,
            "calibration_benign_sessions": 150,
            "suite_benign_sessions": 150,
            "suite_instances_per_variant": 2,
        },
        "detector": {"model": "random_forest", "n_estimators": 40, "max_depth": None,
                     "min_samples_leaf": 2, "class_weight": "balanced",
                     "use_isolation_forest": False},
        "threshold": {"fpr_budget": 0.01},
        "governance": {"recall_tolerance": 0.0, "fpr_tolerance": 0.01,
                       "budget_tolerance": 0.002,
                       "inject_aggressive_candidate_round": 2,
                       "aggressive_fpr_multiplier": 10.0},
        "policy": {"epsilon": 0.2, "seed_rounds": 1},
    }

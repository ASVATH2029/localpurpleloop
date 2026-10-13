"""The governance layer: registry versioning, the gate, rollback."""
from __future__ import annotations

import pytest

from purpleloop.registry import Registry, evaluate_gate

BODY = {"threshold": 0.5, "model_kind": "random_forest"}


def _add(reg, *, round_no, recall, fpr, decision="promoted", promote=True, kind="retrained"):
    return reg.add_version(
        rule_id="test-rule",
        created_round=round_no,
        held_out_recall=recall,
        benign_fpr=fpr,
        suite_recall=recall,
        decision=decision,
        reason="test",
        rule_body=BODY,
        model_path=f"models/v{round_no}.joblib",
        candidate_kind=kind,
        promote=promote,
    )


# ------------------------------------------------------------- registry
def test_first_version_becomes_active(tmp_path):
    reg = Registry(tmp_path / "r.db")
    assert reg.get_active() is None
    v = _add(reg, round_no=0, recall=0.8, fpr=0.01)
    active = reg.get_active()
    assert active["version"] == v
    assert active["rule_body"]["threshold"] == 0.5


def test_only_one_version_is_ever_active(tmp_path):
    reg = Registry(tmp_path / "r.db")
    _add(reg, round_no=0, recall=0.5, fpr=0.01)
    v2 = _add(reg, round_no=1, recall=0.7, fpr=0.01)
    hist = reg.history()
    assert hist["promoted"].sum() == 1
    assert reg.get_active()["version"] == v2


def test_rejected_candidate_does_not_become_active(tmp_path):
    reg = Registry(tmp_path / "r.db")
    v1 = _add(reg, round_no=0, recall=0.5, fpr=0.01)
    _add(reg, round_no=1, recall=0.2, fpr=0.09,
         decision="rejected", promote=False)
    assert reg.get_active()["version"] == v1
    hist = reg.history()
    assert len(hist) == 2
    assert (hist["decision"] == "rejected").sum() == 1


def test_nothing_is_overwritten(tmp_path):
    """Every round leaves a row behind -- that is what buys the audit trail."""
    reg = Registry(tmp_path / "r.db")
    for r in range(5):
        _add(reg, round_no=r, recall=0.1 * r, fpr=0.01)
    hist = reg.history()
    assert len(hist) == 5
    assert list(hist["created_round"]) == [0, 1, 2, 3, 4]


def test_rollback_reactivates_a_previous_version(tmp_path):
    reg = Registry(tmp_path / "r.db")
    v1 = _add(reg, round_no=0, recall=0.5, fpr=0.01)
    v2 = _add(reg, round_no=1, recall=0.9, fpr=0.01)
    assert reg.get_active()["version"] == v2
    reg.rollback(to_version=v1, at_round=2, reason="demo")
    assert reg.get_active()["version"] == v1
    log = reg.rollback_log()
    assert len(log) == 1
    assert int(log.iloc[0]["to_version"]) == v1
    assert int(log.iloc[0]["from_version"]) == v2
    # the newer version still exists, it is simply not active
    assert len(reg.history()) == 2


def test_rollback_to_a_rejected_version_is_refused(tmp_path):
    reg = Registry(tmp_path / "r.db")
    _add(reg, round_no=0, recall=0.5, fpr=0.01)
    bad = _add(reg, round_no=1, recall=0.1, fpr=0.2,
               decision="rejected", promote=False)
    with pytest.raises(ValueError, match="never promoted"):
        reg.rollback(to_version=bad, at_round=2)


def test_rollback_to_unknown_version_raises(tmp_path):
    reg = Registry(tmp_path / "r.db")
    _add(reg, round_no=0, recall=0.5, fpr=0.01)
    with pytest.raises(KeyError):
        reg.rollback(to_version=999, at_round=1)


def test_audit_trail_mentions_every_version_and_rollback(tmp_path):
    reg = Registry(tmp_path / "r.db")
    v1 = _add(reg, round_no=0, recall=0.5, fpr=0.01)
    _add(reg, round_no=1, recall=0.1, fpr=0.2, decision="rejected", promote=False)
    reg.rollback(to_version=v1, at_round=2, reason="demo")
    trail = reg.audit_trail()
    assert len(trail) == 3
    assert any("REJECTED" in line for line in trail)
    assert any("ROLLBACK" in line for line in trail)


def test_bad_decision_value_is_refused(tmp_path):
    reg = Registry(tmp_path / "r.db")
    with pytest.raises(ValueError, match="promoted' or 'rejected"):
        _add(reg, round_no=0, recall=0.5, fpr=0.01, decision="maybe")


# ----------------------------------------------------------------- gate
BUDGET = 0.01


def test_first_version_promoted_when_inside_budget():
    d = evaluate_gate(
        candidate_suite_recall=0.8, candidate_calibrated_fpr=0.009,
        candidate_suite_fpr=0.01, active_suite_recall=None,
        active_suite_fpr=None, fpr_budget=BUDGET,
    )
    assert d.promote and "first version" in d.reason


def test_budget_breach_is_rejected_even_with_better_recall():
    """The whole point: recall bought with false positives is refused."""
    d = evaluate_gate(
        candidate_suite_recall=1.0, candidate_calibrated_fpr=0.10,
        candidate_suite_fpr=0.11, active_suite_recall=0.6,
        active_suite_fpr=0.01, fpr_budget=BUDGET,
    )
    assert not d.promote
    assert "breaches" in d.reason


def test_improvement_on_both_axes_is_promoted():
    d = evaluate_gate(
        candidate_suite_recall=0.9, candidate_calibrated_fpr=0.009,
        candidate_suite_fpr=0.008, active_suite_recall=0.7,
        active_suite_fpr=0.012, fpr_budget=BUDGET,
    )
    assert d.promote and "no regression" in d.reason


def test_recall_regression_is_rejected():
    d = evaluate_gate(
        candidate_suite_recall=0.5, candidate_calibrated_fpr=0.008,
        candidate_suite_fpr=0.005, active_suite_recall=0.9,
        active_suite_fpr=0.01, fpr_budget=BUDGET,
    )
    assert not d.promote and "recall regressed" in d.reason


def test_suite_fpr_regression_beyond_tolerance_is_rejected():
    d = evaluate_gate(
        candidate_suite_recall=0.95, candidate_calibrated_fpr=0.009,
        candidate_suite_fpr=0.05, active_suite_recall=0.9,
        active_suite_fpr=0.01, fpr_budget=BUDGET, fpr_tolerance=0.01,
    )
    assert not d.promote and "FPR regressed" in d.reason


def test_small_fpr_wobble_is_tolerated():
    """Benign sampling noise must not thrash the active version."""
    d = evaluate_gate(
        candidate_suite_recall=0.9, candidate_calibrated_fpr=0.009,
        candidate_suite_fpr=0.015, active_suite_recall=0.9,
        active_suite_fpr=0.010, fpr_budget=BUDGET, fpr_tolerance=0.01,
    )
    assert d.promote


def test_regression_on_both_axes_is_reported_as_such():
    d = evaluate_gate(
        candidate_suite_recall=0.4, candidate_calibrated_fpr=0.009,
        candidate_suite_fpr=0.06, active_suite_recall=0.9,
        active_suite_fpr=0.01, fpr_budget=BUDGET,
    )
    assert not d.promote and "both axes" in d.reason

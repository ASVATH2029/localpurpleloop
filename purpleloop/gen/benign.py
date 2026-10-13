"""Benign background traffic.

Owner: Cloud Specialist (plan Phase 1).

Several kinds are deliberate HARD NEGATIVES -- legitimate activity that looks
like an attack along one axis:

  user_onboarding          CreateUser -> AttachUserPolicy, but ReadOnlyAccess
                           (shape of IAM escalation, without the admin grant)
  backup_job               hundreds of GetObject at a steady rate
                           (volume of exfiltration, on a non-sensitive bucket)
  analyst_sensitive_query  reads from a sensitive bucket, a handful of objects
                           (sensitivity of exfiltration, without the volume)
  log_maintenance          PutEventSelectors / UpdateTrail, broadening scope
                           (shape of trail tampering, opposite intent)

Without these, benign FPR would be free and the 1% budget would mean nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema
from . import common

BENIGN_KINDS = [
    "dev_read",
    "admin_iam_routine",
    "backup_job",
    "ci_deploy",
    "user_onboarding",
    "log_maintenance",
    "analyst_sensitive_query",
    "etl_pipeline",
]

# Roughly a third of benign traffic is a hard negative.
_KIND_WEIGHTS = np.array([0.22, 0.13, 0.10, 0.14, 0.10, 0.07, 0.12, 0.12])

# Kinds that run as overnight batch jobs rather than during business hours.
_NOCTURNAL = {"backup_job", "etl_pipeline"}
_SERVICE_RUN = {"backup_job", "etl_pipeline", "ci_deploy"}


def _dev_read(rng):
    bucket = str(rng.choice(schema.NORMAL_BUCKETS))
    ops = [{"eventName": "ListBucket", "resource": bucket, "params": {}}]
    for _ in range(int(rng.integers(3, 25))):
        ops.append({
            "eventName": "GetObject",
            "resource": f"{bucket}/obj-{int(rng.integers(0, 99999)):05d}",
            "params": {},
        })
    return ops, "fast"


def _admin_iam_routine(rng):
    ops = [{"eventName": "GetCallerIdentity", "resource": "sts", "params": {}}]
    for _ in range(int(rng.integers(3, 14))):
        call = str(rng.choice(["ListUsers", "ListRoles", "GetUser",
                               "ListAttachedUserPolicies", "GetAccountSummary"]))
        who = str(rng.choice(schema.HUMAN_IDENTITIES))
        ops.append({
            "eventName": call,
            "resource": f"user/{who}" if "User" in call else "iam",
            "params": {},
        })
    return ops, "spaced"


def _backup_job(rng):
    # High volume, steady cadence, non-sensitive bucket: volume alone is not
    # evidence of exfiltration.
    bucket = "acme-backups"
    ops = [{"eventName": "ListBucket", "resource": bucket, "params": {}}]
    for _ in range(int(rng.integers(40, 160))):
        ops.append({
            "eventName": "GetObject",
            "resource": f"{bucket}/snap-{int(rng.integers(0, 999999)):06d}.tar",
            "params": {},
        })
    return ops, "burst"


def _ci_deploy(rng):
    ops = [{"eventName": "GetCallerIdentity", "resource": "sts", "params": {}}]
    for _ in range(int(rng.integers(4, 20))):
        if rng.random() < 0.7:
            ops.append({
                "eventName": "PutObject",
                "resource": f"acme-build-artifacts/build-{int(rng.integers(0, 99999)):05d}.zip",
                "params": {},
            })
        else:
            ops.append({
                "eventName": "UpdateFunctionCode",
                "resource": f"function/api-{int(rng.integers(1, 9))}",
                "params": {},
            })
    return ops, "fast"


def _user_onboarding(rng):
    # Same calls as IAM escalation -- but the policy granted is read-only.
    ops = [{"eventName": "ListUsers", "resource": "iam", "params": {}}]
    for _ in range(int(rng.integers(1, 4))):
        who = f"new.hire{int(rng.integers(100, 999))}"
        ops.append({"eventName": "CreateUser", "resource": f"user/{who}", "params": {}})
        ops.append({"eventName": "CreateAccessKey", "resource": f"user/{who}", "params": {}})
        ops.append({
            "eventName": "AttachUserPolicy",
            "resource": f"user/{who}",
            "params": {"policy": str(rng.choice(schema.BENIGN_POLICIES)), "admin": False},
        })
        ops.append({"eventName": "TagUser", "resource": f"user/{who}", "params": {}})
    return ops, str(rng.choice(["fast", "spaced"]))


def _log_maintenance(rng):
    # Trail reconfiguration that widens coverage, not narrows it.
    trail = str(rng.choice(schema.TRAILS))
    ops = [{"eventName": "DescribeTrails", "resource": "cloudtrail", "params": {}}]
    for _ in range(int(rng.integers(1, 4))):
        call = str(rng.choice(["PutEventSelectors", "UpdateTrail"]))
        ops.append({
            "eventName": call,
            "resource": trail,
            "params": {"scope": "broaden", "data_events": True, "multi_region": True},
        })
    return ops, "spaced"


def _analyst_sensitive_query(rng):
    # Authorised access to sensitive data, at human scale.
    bucket = str(rng.choice(schema.SENSITIVE_BUCKETS))
    ops = [{"eventName": "ListBucket", "resource": bucket, "params": {}}]
    for _ in range(int(rng.integers(1, 7))):
        ops.append({
            "eventName": "GetObject",
            "resource": f"{bucket}/report-{int(rng.integers(0, 9999)):04d}.csv",
            "params": {},
        })
    return ops, str(rng.choice(["spaced", "slow"]))


def _etl_pipeline(rng):
    src = "acme-datasets-open"
    ops = []
    for _ in range(int(rng.integers(20, 70))):
        if rng.random() < 0.75:
            ops.append({
                "eventName": "GetObject",
                "resource": f"{src}/part-{int(rng.integers(0, 99999)):05d}.parquet",
                "params": {},
            })
        else:
            ops.append({
                "eventName": "PutObject",
                "resource": f"acme-app-logs/etl-{int(rng.integers(0, 99999)):05d}.json",
                "params": {},
            })
    return ops, "burst"


_BUILDERS = {
    "dev_read": _dev_read,
    "admin_iam_routine": _admin_iam_routine,
    "backup_job": _backup_job,
    "ci_deploy": _ci_deploy,
    "user_onboarding": _user_onboarding,
    "log_maintenance": _log_maintenance,
    "analyst_sensitive_query": _analyst_sensitive_query,
    "etl_pipeline": _etl_pipeline,
}


def generate_benign(
    n_sessions: int,
    seed: int,
    *,
    round_no: int = 0,
    day: int | None = None,
) -> pd.DataFrame:
    """Generate n_sessions of benign background traffic."""
    rng = np.random.default_rng(seed)
    day = round_no if day is None else day
    frames = []

    for _ in range(n_sessions):
        kind = str(rng.choice(BENIGN_KINDS, p=_KIND_WEIGHTS))
        ops, profile = _BUILDERS[kind](rng)
        if not ops:
            continue
        pool = schema.SERVICE_IDENTITIES if kind in _SERVICE_RUN else schema.HUMAN_IDENTITIES
        start = (
            common.day_start(day) + rng.uniform(0, 6 * 3600)
            if kind in _NOCTURNAL
            else common.business_hours_start(rng, day)
        )
        frames.append(common.ops_to_events(
            ops,
            start_ts=start,
            gaps=common.delays(rng, profile, len(ops) - 1),
            identities=common.assign_identities(rng, len(ops), "single", pool=pool),
            session=common.session_id(rng, "ben"),
            round_no=round_no,
            variant=f"benign:{kind}",
            label=0,
            ip=common.source_ip(rng, external=False),
        ))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=schema.EVENT_COLUMNS
    )
    common.validate_events(out)
    return out

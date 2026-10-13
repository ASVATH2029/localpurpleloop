"""The three seed attack families and the mutation knob space.

Owner: Cybersecurity Analyst (plan Phase 1).

Each family has a canonical op sequence that expresses the *technique*. The
four knobs then mutate how that technique is delivered -- timing, volume,
ordering, and how many identities it is spread across -- without changing what
it accomplishes. That split is the whole point: a detector that keys on the
technique generalises, one that keys on the delivery does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema
from . import common

# --------------------------------------------------------------------------
# Canonical op sequences, one per family
# --------------------------------------------------------------------------
def _iam_escalation_ops(rng: np.random.Generator, n_units: int) -> list[dict]:
    """CreateUser -> (CreateAccessKey) -> AttachUserPolicy(AdministratorAccess).

    The technique signal is granting an admin-equivalent policy. Benign
    onboarding runs the same calls but grants ReadOnlyAccess.
    """
    ops: list[dict] = [
        {"eventName": "GetCallerIdentity", "resource": "sts", "params": {}},
        {"eventName": "ListUsers", "resource": "iam", "params": {}},
    ]
    for i in range(n_units):
        victim = f"svc-tmp-{int(rng.integers(1000, 9999))}"
        ops.append({"eventName": "CreateUser", "resource": f"user/{victim}", "params": {}})
        if rng.random() < 0.6:
            ops.append({
                "eventName": "CreateAccessKey",
                "resource": f"user/{victim}",
                "params": {},
            })
        policy = str(rng.choice(schema.ADMIN_POLICIES))
        grant_call = "PutUserPolicy" if policy.startswith("inline:") else "AttachUserPolicy"
        ops.append({
            "eventName": grant_call,
            "resource": f"user/{victim}",
            "params": {"policy": policy, "admin": True},
        })
        if rng.random() < 0.25:
            ops.append({
                "eventName": "ListAttachedUserPolicies",
                "resource": f"user/{victim}",
                "params": {},
            })
    return ops


def _s3_exfiltration_ops(rng: np.random.Generator, n_reads: int) -> list[dict]:
    """Recon a sensitive bucket, then pull objects out of it.

    The technique signal is read volume concentrated on sensitive data by one
    principal. Benign analysts read the same buckets, but a handful of objects.
    """
    bucket = str(rng.choice(schema.SENSITIVE_BUCKETS))
    ops: list[dict] = [
        {"eventName": "ListBucket", "resource": bucket, "params": {}},
    ]
    if rng.random() < 0.5:
        ops.append({"eventName": "GetBucketAcl", "resource": bucket, "params": {}})
    if rng.random() < 0.3:
        ops.append({"eventName": "GetBucketPolicy", "resource": bucket, "params": {}})
    for _ in range(n_reads):
        ops.append({
            "eventName": "GetObject",
            "resource": f"{bucket}/export-{int(rng.integers(0, 999999)):06d}.csv",
            "params": {},
        })
    return ops


def _cloudtrail_tampering_ops(rng: np.random.Generator, n_ops: int) -> list[dict]:
    """Stop / delete a trail, or quietly narrow what it records.

    The technique signal is reducing audit coverage. Benign maintenance also
    calls PutEventSelectors, but to broaden scope, never to narrow it.
    """
    trail = str(rng.choice(schema.TRAILS))
    ops: list[dict] = [
        {"eventName": "DescribeTrails", "resource": "cloudtrail", "params": {}},
    ]
    choices = ["StopLogging", "PutEventSelectors", "DeleteTrail", "DeleteLogGroup", "UpdateTrail"]
    weights = [0.34, 0.3, 0.16, 0.1, 0.1]
    for _ in range(n_ops):
        call = str(rng.choice(choices, p=weights))
        if call == "PutEventSelectors":
            ops.append({
                "eventName": call,
                "resource": trail,
                "params": {"scope": "narrow", "data_events": False},
            })
        elif call == "UpdateTrail":
            ops.append({
                "eventName": call,
                "resource": trail,
                "params": {"scope": "narrow", "multi_region": False},
            })
        elif call == "DeleteLogGroup":
            ops.append({
                "eventName": call,
                "resource": "log-group/cloudtrail-audit",
                "params": {},
            })
        else:
            ops.append({"eventName": call, "resource": trail, "params": {}})
    return ops


_FAMILY_BUILDERS = {
    "iam_escalation": _iam_escalation_ops,
    "s3_exfiltration": _s3_exfiltration_ops,
    "cloudtrail_tampering": _cloudtrail_tampering_ops,
}


# --------------------------------------------------------------------------
# Knob application
# --------------------------------------------------------------------------
def _apply_order(rng: np.random.Generator, ops: list[dict], order: str) -> list[dict]:
    """Rearrange / dilute the op sequence per the call_order knob."""
    if order == "canonical":
        return ops
    if order == "reversed":
        # Breaks the canonical bigrams (e.g. identity_create -> privilege_grant).
        return list(reversed(ops))
    if order == "interleaved":
        out: list[dict] = []
        for op in ops:
            out.append(op)
            out.extend(common.filler_ops(rng, int(rng.integers(1, 4))))
        return out
    if order == "split":
        # Break the core into chunks separated by blocks of innocuous traffic.
        n_chunks = int(rng.integers(2, 4))
        chunks = np.array_split(np.arange(len(ops)), n_chunks)
        out = common.filler_ops(rng, int(rng.integers(2, 6)))
        for ci, chunk in enumerate(chunks):
            out.extend(ops[i] for i in chunk)
            if ci < len(chunks) - 1:
                out.extend(common.filler_ops(rng, int(rng.integers(3, 9))))
        out.extend(common.filler_ops(rng, int(rng.integers(2, 6))))
        return out
    raise ValueError(f"unknown call_order: {order}")


def sample_knobs(
    rng: np.random.Generator,
    bias: dict | None = None,
    epsilon: float = 0.2,
    exclude: set[str] | None = None,
    family: str | None = None,
    max_tries: int = 60,
) -> tuple[str, dict]:
    """Draw a (family, knobs) pair, optionally pinning specific knob values.

    bias is a {knob: value} mapping from the mutation policy -- the exact knob
    value whose misses it wants to probe again. With probability epsilon the
    bias is ignored, so the loop keeps exploring and a pinned value cannot
    starve the rest of the space. exclude holds variant_ids already seen, so
    each round gets a genuinely new delivery.

    Pinning the value, not merely the dimension, is what lets the policy's own
    hypothesis be re-tested: if the retrained detector now catches that value,
    its miss estimate falls and the policy moves on by itself. Biasing only the
    dimension and then drawing some other value within it never gathers
    evidence about the value that triggered the bias, so the estimate goes
    stale and the policy sticks.
    """
    exclude = exclude or set()
    honour_bias = bool(bias) and rng.random() > epsilon
    for _ in range(max_tries):
        fam = family or str(rng.choice(schema.FAMILIES))
        knobs = {k: str(rng.choice(v)) for k, v in schema.KNOBS.items()}
        if honour_bias:
            for dim, value in bias.items():
                if value in schema.KNOBS.get(dim, []):
                    knobs[dim] = str(value)
        vid = schema.variant_id(fam, knobs)
        if vid not in exclude:
            return fam, knobs
    # This corner of the space is exhausted: drop the bias rather than stall.
    for _ in range(max_tries):
        fam = family or str(rng.choice(schema.FAMILIES))
        knobs = {k: str(rng.choice(v)) for k, v in schema.KNOBS.items()}
        if schema.variant_id(fam, knobs) not in exclude:
            return fam, knobs
    return fam, knobs


def seed_variant_ids() -> list[str]:
    """variant_ids of the hand-authored seed variants."""
    return [
        schema.variant_id(fam, knobs)
        for fam, variants in schema.SEED_VARIANTS.items()
        for knobs in variants
    ]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def generate_variant(
    family: str,
    knobs: dict,
    seed: int,
    *,
    n_sessions: int = 1,
    round_no: int = 0,
    day: int | None = None,
) -> pd.DataFrame:
    """Generate n_sessions independent instances of one attack variant.

    Each session is a separate episode (its own session_id), so held-out recall
    is a fraction over several attempts rather than a single coin flip.
    """
    if family not in _FAMILY_BUILDERS:
        raise ValueError(f"unknown family: {family}")
    missing = set(schema.KNOB_NAMES) - set(knobs)
    if missing:
        raise ValueError(f"knobs missing: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    day = round_no if day is None else day
    vid = schema.variant_id(family, knobs)
    frames = []

    for _ in range(n_sessions):
        n = common.batch_count(rng, knobs["batch_size"])
        core = _FAMILY_BUILDERS[family](rng, n)
        ops = _apply_order(rng, core, knobs["call_order"])
        gaps = common.delays(rng, knobs["delay_profile"], len(ops) - 1)
        idents = common.assign_identities(
            rng, len(ops), knobs["identity_reuse"], pool=schema.ALL_IDENTITIES
        )
        frames.append(common.ops_to_events(
            ops,
            start_ts=common.uniform_day_start(rng, day),
            gaps=gaps,
            identities=idents,
            session=common.session_id(rng, "atk"),
            round_no=round_no,
            variant=vid,
            label=1,
            ip=common.source_ip(rng, external=bool(rng.random() < 0.5)),
        ))

    out = pd.concat(frames, ignore_index=True)
    common.validate_events(out)
    return out

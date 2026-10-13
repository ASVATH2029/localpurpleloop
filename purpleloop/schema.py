"""Frozen shared contracts: event schema, API vocabulary, semantics, knobs.

Everything downstream (generators, features, model, registry) codes against
this module. Phase 0 of the plan freezes it; later phases must not change it.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Event schema (one row = one simulated control-plane API call)
# --------------------------------------------------------------------------
EVENT_COLUMNS = [
    "timestamp",    # float, epoch seconds
    "identity",     # principal that made the call
    "session_id",   # episode grouping key -- the unit of detection
    "eventName",    # API call name
    "resource",     # bucket / user / trail touched
    "source_ip",
    "params",       # JSON string, call-specific detail (policy granted, scope, ...)
    "round",        # which loop round produced it
    "variant_id",   # attack variant id, or "benign:<kind>"
    "label",        # 1 = attack, 0 = benign
]

# --------------------------------------------------------------------------
# API vocabulary and semantic categories
# --------------------------------------------------------------------------
# Categories are what let the detector generalise across timing mutations:
# a mutated attack changes WHEN calls happen, not WHAT they mean.
CATEGORY_OF = {
    # identity creation
    "CreateUser": "identity_create",
    "CreateAccessKey": "identity_create",
    "CreateRole": "identity_create",
    # privilege granting
    "AttachUserPolicy": "privilege_grant",
    "AttachRolePolicy": "privilege_grant",
    "PutUserPolicy": "privilege_grant",
    "CreatePolicyVersion": "privilege_grant",
    "UpdateAssumeRolePolicy": "privilege_grant",
    # audit-log interference
    "StopLogging": "logging_disable",
    "DeleteTrail": "logging_disable",
    "DeleteLogGroup": "logging_disable",
    "PutEventSelectors": "log_scope",
    "UpdateTrail": "log_scope",
    # data plane
    "GetObject": "object_read",
    "PutObject": "object_write",
    "DeleteObject": "object_write",
    # reconnaissance / read-only
    "ListBucket": "recon",
    "ListUsers": "recon",
    "ListRoles": "recon",
    "GetUser": "recon",
    "GetBucketAcl": "recon",
    "GetBucketPolicy": "recon",
    "DescribeTrails": "recon",
    "GetCallerIdentity": "recon",
    "ListAttachedUserPolicies": "recon",
    # other
    "UpdateFunctionCode": "compute",
    "InvokeFunction": "compute",
    "TagUser": "misc",
    "GetAccountSummary": "misc",
}

API_VOCAB = sorted(CATEGORY_OF)
CATEGORIES = sorted(set(CATEGORY_OF.values()))

WRITE_CATEGORIES = {
    "identity_create", "privilege_grant", "logging_disable",
    "log_scope", "object_write", "compute",
}
READ_CATEGORIES = {"object_read", "recon", "misc"}

# --------------------------------------------------------------------------
# Resources. Sensitivity is attacker-relevant domain knowledge, not a label:
# benign traffic also touches sensitive buckets (authorised analysts), so mere
# presence proves nothing -- volume and concentration are what matter.
# --------------------------------------------------------------------------
SENSITIVE_BUCKETS = [
    "acme-customer-pii",
    "acme-payroll-exports",
    "acme-secrets-store",
    "acme-phi-archive",
]
NORMAL_BUCKETS = [
    "acme-public-assets",
    "acme-web-static",
    "acme-app-logs",
    "acme-backups",
    "acme-build-artifacts",
    "acme-datasets-open",
]
ALL_BUCKETS = SENSITIVE_BUCKETS + NORMAL_BUCKETS
TRAILS = ["trail/org-main", "trail/security-audit", "trail/data-events"]

# Policies. The admin / non-admin split is the real escalation signal: benign
# onboarding also runs CreateUser -> AttachUserPolicy, but grants ReadOnlyAccess.
ADMIN_POLICIES = [
    "arn:aws:iam::aws:policy/AdministratorAccess",
    "arn:aws:iam::aws:policy/PowerUserAccess",
    "inline:AllowStar",            # Action "*" on Resource "*"
]
BENIGN_POLICIES = [
    "arn:aws:iam::aws:policy/ReadOnlyAccess",
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
    "arn:aws:iam::aws:policy/job-function/SupportUser",
    "arn:aws:iam::aws:policy/job-function/Billing",
]

HUMAN_IDENTITIES = [
    "alice.dev", "bob.dev", "carol.analyst", "dan.analyst",
    "erin.admin", "frank.admin", "grace.sre",
]
SERVICE_IDENTITIES = [
    "svc-ci-deploy", "svc-backup", "svc-etl", "svc-monitoring",
]
ALL_IDENTITIES = HUMAN_IDENTITIES + SERVICE_IDENTITIES

# --------------------------------------------------------------------------
# Attack families and the mutation knob space
# --------------------------------------------------------------------------
FAMILIES = ["iam_escalation", "s3_exfiltration", "cloudtrail_tampering"]

KNOBS = {
    # inter-call delay: burst is the obvious case, slow/jittered are evasive
    "delay_profile": ["burst", "fast", "spaced", "slow", "jittered"],
    "batch_size":    ["small", "medium", "large"],
    "call_order":    ["canonical", "reversed", "interleaved", "split"],
    "identity_reuse": ["single", "two_identities", "per_call_identity"],
}
KNOB_NAMES = list(KNOBS)

# Seed variants (plan Phase 1: 2-3 hand-authored per family). Deliberately
# NARROW -- all burst/fast, canonical, single identity -- so the first models
# lean on timing and the generalisation curve has somewhere to climb from.
SEED_VARIANTS = {
    "iam_escalation": [
        {"delay_profile": "burst", "batch_size": "small",  "call_order": "canonical", "identity_reuse": "single"},
        {"delay_profile": "burst", "batch_size": "medium", "call_order": "canonical", "identity_reuse": "single"},
    ],
    "s3_exfiltration": [
        {"delay_profile": "burst", "batch_size": "medium", "call_order": "canonical", "identity_reuse": "single"},
        {"delay_profile": "fast",  "batch_size": "large",  "call_order": "canonical", "identity_reuse": "single"},
    ],
    "cloudtrail_tampering": [
        {"delay_profile": "burst", "batch_size": "small",  "call_order": "canonical", "identity_reuse": "single"},
        {"delay_profile": "fast",  "batch_size": "small",  "call_order": "canonical", "identity_reuse": "single"},
    ],
}

# Delay profiles -> (low, high) seconds between consecutive calls
DELAY_RANGES = {
    "burst":    (0.05, 0.6),
    "fast":     (1.0, 4.0),
    "spaced":   (20.0, 120.0),
    "slow":     (240.0, 900.0),
    "jittered": (0.1, 600.0),   # heavy-tailed, drawn lognormal
}
BATCH_SIZES = {"small": (2, 4), "medium": (5, 12), "large": (15, 40)}


def variant_id(family: str, knobs: dict) -> str:
    """Stable, human-readable id: family plus a knob fingerprint."""
    return "{}|{}".format(
        family, ",".join(f"{k}={knobs[k]}" for k in KNOB_NAMES)
    )


def parse_variant_id(vid: str) -> tuple[str, dict]:
    """Inverse of variant_id. Returns (family, knobs)."""
    family, _, knob_part = vid.partition("|")
    knobs = {}
    if knob_part:
        for piece in knob_part.split(","):
            k, _, v = piece.partition("=")
            knobs[k] = v
    return family, knobs


def is_sensitive(resource: str) -> bool:
    """True if the resource lives in a sensitive bucket."""
    return resource.split("/")[0] in SENSITIVE_BUCKETS

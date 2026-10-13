"""Session -> fixed-length feature vector.

Owner: ML Engineer #1, reviewed by the Cybersecurity Analyst (plan Phase 2).

The feature set is split on purpose into two kinds:

  DELIVERY features   timing, volume, burstiness, directed call bigrams.
                      These describe HOW an attack was delivered and so they
                      break under timing / ordering mutation.

  TECHNIQUE features  admin-policy grants, log-coverage reduction, read volume
                      concentrated on sensitive data, unordered co-occurrence
                      of categories. These describe WHAT the attack achieves
                      and survive mutation.

Early rounds train on narrow seed variants where both kinds separate the
classes equally well, so the forest leans partly on delivery and misses
mutated variants. As mutated variants accumulate, the delivery features stop
being pure and the technique features take over. That shift is the mechanism
behind the generalisation curve -- it is not special-cased anywhere.

Nothing here may read label, round, variant_id or session_id: that would leak.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import schema

META_COLUMNS = ["session_id", "label", "round", "variant_id"]

# Directed category bigrams (delivery-ish: order sensitive)
BIGRAMS = [
    ("identity_create", "privilege_grant"),
    ("privilege_grant", "logging_disable"),
    ("recon", "object_read"),
    ("object_read", "object_read"),
    ("recon", "privilege_grant"),
    ("logging_disable", "logging_disable"),
    ("recon", "log_scope"),
    ("identity_create", "identity_create"),
]

# Unordered category co-occurrence (technique-ish: order invariant)
CO_PAIRS = [
    ("identity_create", "privilege_grant"),
    ("privilege_grant", "logging_disable"),
    ("recon", "logging_disable"),
    ("recon", "object_read"),
]


def _build_feature_names() -> list[str]:
    names = [
        # volume / duration
        "n_events", "log_n_events", "duration_s", "log_duration_s", "events_per_min",
        # timing
        "iat_mean", "iat_median", "iat_std", "iat_cv", "iat_min", "iat_max",
        "iat_entropy", "burst_ratio_60s",
        # diversity
        "n_distinct_api", "n_distinct_resources", "n_distinct_identities",
        "n_distinct_ips", "resource_concentration", "bucket_concentration",
        # semantic / technique
        "n_admin_grant", "admin_grant_present", "n_privilege_grant",
        "n_identity_create", "n_logging_disable", "n_log_narrow", "n_log_broaden",
        "n_sensitive_reads", "sensitive_read_frac", "sensitive_bucket_touched",
        "max_reads_one_sensitive_bucket", "write_frac", "read_frac", "external_ip",
        # time of day
        "hour_sin", "hour_cos", "is_off_hours", "day_of_week",
    ]
    names += [f"frac_{c}" for c in schema.CATEGORIES]
    names += [f"logcount_{c}" for c in schema.CATEGORIES]
    names += [f"bigram_{a}__{b}" for a, b in BIGRAMS]
    names += [f"cooc_{a}__{b}" for a, b in CO_PAIRS]
    return names


FEATURE_NAMES = _build_feature_names()


def _build_feature_groups() -> dict[str, list[str]]:
    """Split the feature set into DELIVERY ("how") and TECHNIQUE ("what").

    Used by the ablation: a detector restricted to delivery features can only
    recognise attacks delivered the way it has already seen, which is the
    failure mode the whole project is about.
    """
    # What the session ACHIEVES: which calls were made, on what kind of data,
    # with what effect on privilege and audit coverage. Mutating timing,
    # ordering or identity spread does not change these.
    technique_exact = {
        "n_admin_grant", "admin_grant_present", "n_privilege_grant",
        "n_identity_create", "n_logging_disable", "n_log_narrow", "n_log_broaden",
        "n_sensitive_reads", "sensitive_read_frac", "sensitive_bucket_touched",
        "max_reads_one_sensitive_bucket", "write_frac", "read_frac",
    }
    # frac_<category> and logcount_<category> both say WHICH calls were made,
    # so they belong to technique. Leaving them on the delivery side makes the
    # delivery arm secretly semantic and the ablation unable to separate the
    # two groups at all.
    technique = [
        f for f in FEATURE_NAMES
        if f in technique_exact or f.startswith(("frac_", "logcount_", "cooc_"))
    ]
    # Everything else: HOW it was delivered -- timing, burstiness, duration,
    # volume, diversity, concentration, time of day, and the order-sensitive
    # directed bigrams.
    delivery = [f for f in FEATURE_NAMES if f not in set(technique)]
    assert len(technique) + len(delivery) == len(FEATURE_NAMES)
    return {
        "all": list(FEATURE_NAMES),
        "delivery_only": delivery,
        "technique_only": technique,
    }


FEATURE_GROUPS = _build_feature_groups()
DELIVERY_FEATURES = FEATURE_GROUPS["delivery_only"]
TECHNIQUE_FEATURES = FEATURE_GROUPS["technique_only"]


def _entropy(x: np.ndarray) -> float:
    """Normalised Shannon entropy of a gap distribution (1 = perfectly even)."""
    total = float(x.sum())
    if x.size < 2 or total <= 0:
        return 0.0
    p = x / total
    p = p[p > 0]
    if p.size < 2:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def _burst_ratio(ts: np.ndarray, window: float = 60.0) -> float:
    """Largest share of the session's calls falling inside any 60s window."""
    n = ts.size
    if n <= 1:
        return 1.0
    right = np.searchsorted(ts, ts + window, side="right")
    return float((right - np.arange(n)).max() / n)


def _session_features(g: pd.DataFrame) -> dict:
    g = g.sort_values("timestamp", kind="stable")
    ts = g["timestamp"].to_numpy(dtype=float)
    names = g["eventName"].to_numpy()
    resources = g["resource"].to_numpy()
    n = len(g)

    cats = np.array([schema.CATEGORY_OF.get(e, "misc") for e in names])
    params = [json.loads(p) if p else {} for p in g["params"].to_numpy()]

    iats = np.diff(ts) if n > 1 else np.zeros(0)
    duration = float(ts[-1] - ts[0]) if n > 1 else 0.0

    buckets = np.array([str(r).split("/")[0] for r in resources])
    is_sens_read = np.array([
        cats[i] == "object_read" and buckets[i] in schema.SENSITIVE_BUCKETS
        for i in range(n)
    ])

    # reads concentrated on a single sensitive bucket -> the exfil conjunction
    # (volume AND sensitivity); a backup job has volume but no sensitivity,
    # an analyst query has sensitivity but no volume.
    max_sens_reads = 0
    if is_sens_read.any():
        vals, counts = np.unique(buckets[is_sens_read], return_counts=True)
        max_sens_reads = int(counts.max())

    _, res_counts = np.unique(resources, return_counts=True)
    _, bkt_counts = np.unique(buckets, return_counts=True)

    admin_grants = sum(1 for p in params if p.get("admin") is True)
    narrow = sum(1 for p in params if p.get("scope") == "narrow")
    broaden = sum(1 for p in params if p.get("scope") == "broaden")

    gm = time.gmtime(ts[0])
    hour = gm.tm_hour + gm.tm_min / 60.0

    feats = {
        "n_events": float(n),
        "log_n_events": float(np.log1p(n)),
        "duration_s": duration,
        "log_duration_s": float(np.log1p(duration)),
        "events_per_min": float(n / (duration / 60.0)) if duration > 0 else float(n),
        "iat_mean": float(iats.mean()) if iats.size else 0.0,
        "iat_median": float(np.median(iats)) if iats.size else 0.0,
        "iat_std": float(iats.std()) if iats.size else 0.0,
        "iat_cv": float(iats.std() / iats.mean()) if iats.size and iats.mean() > 0 else 0.0,
        "iat_min": float(iats.min()) if iats.size else 0.0,
        "iat_max": float(iats.max()) if iats.size else 0.0,
        "iat_entropy": _entropy(iats),
        "burst_ratio_60s": _burst_ratio(ts),
        "n_distinct_api": float(len(set(names))),
        "n_distinct_resources": float(len(set(resources))),
        "n_distinct_identities": float(g["identity"].nunique()),
        "n_distinct_ips": float(g["source_ip"].nunique()),
        "resource_concentration": float(res_counts.max() / n),
        "bucket_concentration": float(bkt_counts.max() / n),
        "n_admin_grant": float(admin_grants),
        "admin_grant_present": 1.0 if admin_grants else 0.0,
        "n_privilege_grant": float((cats == "privilege_grant").sum()),
        "n_identity_create": float((cats == "identity_create").sum()),
        "n_logging_disable": float((cats == "logging_disable").sum()),
        "n_log_narrow": float(narrow),
        "n_log_broaden": float(broaden),
        "n_sensitive_reads": float(is_sens_read.sum()),
        "sensitive_read_frac": float(is_sens_read.sum() / n),
        "sensitive_bucket_touched": 1.0 if any(
            b in schema.SENSITIVE_BUCKETS for b in buckets
        ) else 0.0,
        "max_reads_one_sensitive_bucket": float(max_sens_reads),
        "write_frac": float(np.isin(cats, list(schema.WRITE_CATEGORIES)).sum() / n),
        "read_frac": float(np.isin(cats, list(schema.READ_CATEGORIES)).sum() / n),
        "external_ip": 1.0 if str(g["source_ip"].iloc[0]).startswith("203.") else 0.0,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "is_off_hours": 1.0 if (hour < 7 or hour >= 19) else 0.0,
        "day_of_week": float(gm.tm_wday),
    }

    for c in schema.CATEGORIES:
        k = int((cats == c).sum())
        feats[f"frac_{c}"] = float(k / n)
        feats[f"logcount_{c}"] = float(np.log1p(k))

    denom = max(1, n - 1)
    for a, b in BIGRAMS:
        hits = int(np.sum((cats[:-1] == a) & (cats[1:] == b))) if n > 1 else 0
        feats[f"bigram_{a}__{b}"] = float(hits / denom)

    present = set(cats.tolist())
    for a, b in CO_PAIRS:
        feats[f"cooc_{a}__{b}"] = 1.0 if (a in present and b in present) else 0.0

    return feats


def extract_features(events: pd.DataFrame) -> pd.DataFrame:
    """One row per session: FEATURE_NAMES plus META_COLUMNS.

    Deterministic and stateless -- the same events always give the same rows,
    which is what lets round N's model be compared with round N+1's.
    """
    if events.empty:
        return pd.DataFrame(columns=FEATURE_NAMES + META_COLUMNS)

    rows = []
    for sid, g in events.groupby("session_id", sort=False):
        feats = _session_features(g)
        feats["session_id"] = sid
        feats["label"] = int(g["label"].iloc[0])
        feats["round"] = int(g["round"].iloc[0])
        feats["variant_id"] = str(g["variant_id"].iloc[0])
        rows.append(feats)

    out = pd.DataFrame(rows)
    # Fixed column order, so stored rounds stay mutually comparable.
    return out.reindex(columns=FEATURE_NAMES + META_COLUMNS)


def feature_matrix(df: pd.DataFrame, names: list[str] | None = None) -> np.ndarray:
    """Model input: features only, in the frozen order, NaNs zeroed.

    names restricts the matrix to one feature group (see FEATURE_GROUPS); the
    detector passes the group it was trained on so scoring always matches.
    """
    names = list(names) if names else list(FEATURE_NAMES)
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"feature frame missing columns: {sorted(missing)}")
    return (
        df.reindex(columns=names)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

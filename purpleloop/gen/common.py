"""Shared machinery for both generators: a simulated clock, delay profiles,
identity assignment and the op-list -> DataFrame conversion.

An "op" is a dict with keys eventName / resource / params. Generators build a
list of ops, then this module stamps timestamps, identities and IPs onto them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import schema

# Fixed simulated epoch so runs are comparable: 2025-01-06 00:00:00 UTC (a Monday).
SIM_EPOCH = 1736121600.0
DAY = 86400.0


def day_start(day: int) -> float:
    return SIM_EPOCH + day * DAY


def business_hours_start(rng: np.random.Generator, day: int) -> float:
    """A start time weighted to 09:00-18:00 on weekdays."""
    hour = float(np.clip(rng.normal(13.0, 2.6), 7.0, 20.0))
    return day_start(day) + hour * 3600.0 + rng.uniform(0, 3600)


def uniform_day_start(rng: np.random.Generator, day: int) -> float:
    """A start time uniform across the whole day.

    Attacks use this, so "off hours" is weak evidence rather than a giveaway.
    """
    return day_start(day) + rng.uniform(0, DAY)


def session_id(rng: np.random.Generator, prefix: str) -> str:
    return f"{prefix}-{int(rng.integers(0, 2**31)):010d}"


def source_ip(rng: np.random.Generator, external: bool = False) -> str:
    if external:
        return f"203.0.113.{int(rng.integers(2, 254))}"
    return f"10.{int(rng.integers(0, 32))}.{int(rng.integers(0, 255))}.{int(rng.integers(2, 254))}"


def delays(rng: np.random.Generator, profile: str, n_gaps: int) -> np.ndarray:
    """Inter-call gaps in seconds for a given delay profile."""
    if n_gaps <= 0:
        return np.zeros(0)
    low, high = schema.DELAY_RANGES[profile]
    if profile == "jittered":
        # Heavy-tailed: most gaps short, occasional long pause. This is the
        # profile that defeats fixed-window rules.
        raw = np.exp(rng.normal(np.log(2.0), 1.8, size=n_gaps))
        return np.clip(raw, low, high)
    return rng.uniform(low, high, size=n_gaps)


def batch_count(rng: np.random.Generator, size_name: str) -> int:
    low, high = schema.BATCH_SIZES[size_name]
    return int(rng.integers(low, high + 1))


def filler_ops(rng: np.random.Generator, n: int) -> list[dict]:
    """Innocuous-looking calls used to dilute an attack sequence."""
    out = []
    for _ in range(max(0, n)):
        pick = rng.random()
        if pick < 0.45:
            bucket = str(rng.choice(schema.NORMAL_BUCKETS))
            out.append({
                "eventName": "GetObject",
                "resource": f"{bucket}/obj-{int(rng.integers(0, 99999)):05d}",
                "params": {},
            })
        elif pick < 0.65:
            out.append({
                "eventName": "ListBucket",
                "resource": str(rng.choice(schema.NORMAL_BUCKETS)),
                "params": {},
            })
        elif pick < 0.8:
            out.append({"eventName": "GetCallerIdentity", "resource": "sts", "params": {}})
        elif pick < 0.9:
            out.append({
                "eventName": "GetUser",
                "resource": f"user/{rng.choice(schema.HUMAN_IDENTITIES)}",
                "params": {},
            })
        else:
            out.append({"eventName": "GetAccountSummary", "resource": "account", "params": {}})
    return out


def assign_identities(
    rng: np.random.Generator, n: int, mode: str, pool: list[str] | None = None
) -> list[str]:
    """Spread n calls across identities according to the identity_reuse knob."""
    pool = pool or schema.ALL_IDENTITIES
    if mode == "single":
        who = str(rng.choice(pool))
        return [who] * n
    if mode == "two_identities":
        pair = [str(x) for x in rng.choice(pool, size=min(2, len(pool)), replace=False)]
        return [pair[int(rng.integers(0, len(pair)))] for _ in range(n)]
    # per_call_identity
    return [str(rng.choice(pool)) for _ in range(n)]


def ops_to_events(
    ops: list[dict],
    *,
    start_ts: float,
    gaps: np.ndarray,
    identities: list[str],
    session: str,
    round_no: int,
    variant: str,
    label: int,
    ip: str,
) -> pd.DataFrame:
    """Stamp an op list into a DataFrame matching schema.EVENT_COLUMNS."""
    n = len(ops)
    if n == 0:
        return pd.DataFrame(columns=schema.EVENT_COLUMNS)
    ts = np.empty(n, dtype=float)
    ts[0] = start_ts
    if n > 1:
        ts[1:] = start_ts + np.cumsum(gaps[: n - 1])
    rows = {
        "timestamp": ts,
        "identity": identities[:n],
        "session_id": [session] * n,
        "eventName": [o["eventName"] for o in ops],
        "resource": [o["resource"] for o in ops],
        "source_ip": [ip] * n,
        "params": [json.dumps(o.get("params", {}), sort_keys=True) for o in ops],
        "round": np.full(n, round_no, dtype=int),
        "variant_id": [variant] * n,
        "label": np.full(n, label, dtype=int),
    }
    return pd.DataFrame(rows, columns=schema.EVENT_COLUMNS)


def validate_events(df: pd.DataFrame) -> None:
    """Cheap sanity contract check -- generators call this before returning."""
    missing = set(schema.EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"event frame missing columns: {sorted(missing)}")
    unknown = set(df["eventName"].unique()) - set(schema.API_VOCAB)
    if unknown:
        raise ValueError(f"eventName outside the frozen vocabulary: {sorted(unknown)}")

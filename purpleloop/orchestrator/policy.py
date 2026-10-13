"""The mutation policy: aim the next round at the loop's own weak spot.

Owner: ML Engineer #2, with miss attribution defined by the Cybersecurity
Analyst (plan Phase 4).

This is what makes the loop adaptive rather than a random batch job. After
each round the miss rate of the round's variant is attributed to the four knob
values that produced it. The policy then picks the knob DIMENSION whose worst
value has the highest average miss rate, and the generator biases the next
round toward that dimension's evasive values.

Deliberately a transparent heuristic, not RL: every decision it makes has to
be explainable in one logged sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import KNOB_NAMES

# Shrinkage weight for the per-knob-value miss estimate. A knob value seen in
# a single round would otherwise dominate the policy on one noisy observation,
# which is exactly what happened before this was added: one round that missed
# a medium-batch variant pinned the policy to batch_size for the rest of the
# run while the genuinely weak knob (slow timing, 5 rounds of evidence) was
# ignored. Each value is pulled PRIOR_WEIGHT rounds' worth toward the overall
# miss rate, so evidence has to accumulate before it steers anything.
PRIOR_WEIGHT = 2.0


@dataclass
class MissHistory:
    """Per-round record of which knob values were missed, and how badly."""

    records: list[dict] = field(default_factory=list)

    def add(self, *, round_no: int, family: str, knobs: dict, miss_rate: float) -> None:
        self.records.append({
            "round": int(round_no),
            "family": family,
            "miss_rate": float(miss_rate),
            **{k: knobs.get(k, "?") for k in KNOB_NAMES},
        })

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def knob_miss_table(self) -> pd.DataFrame:
        """Miss rate per (knob dimension, value) seen so far.

        mean_miss_rate is the raw observation; smoothed_miss_rate shrinks it
        toward the overall miss rate by PRIOR_WEIGHT rounds and is what the
        policy actually ranks on.
        """
        df = self.table()
        if df.empty:
            return pd.DataFrame(columns=[
                "knob", "value", "mean_miss_rate", "smoothed_miss_rate", "rounds"
            ])
        global_mean = float(df["miss_rate"].mean())
        rows = []
        for dim in KNOB_NAMES:
            for value, series in df.groupby(dim)["miss_rate"]:
                n = int(series.size)
                rows.append({
                    "knob": dim,
                    "value": value,
                    "mean_miss_rate": float(series.mean()),
                    "smoothed_miss_rate": float(
                        (series.sum() + PRIOR_WEIGHT * global_mean) / (n + PRIOR_WEIGHT)
                    ),
                    "rounds": n,
                })
        return pd.DataFrame(rows).sort_values(
            ["smoothed_miss_rate", "rounds"], ascending=[False, False]
        ).reset_index(drop=True)

    def choose_bias(self) -> tuple[dict | None, str]:
        """Pick the knob value to probe next, plus a one-line justification.

        Returns ({knob: value}, reason). The value is the one whose smoothed
        miss estimate is worst, so the next round re-tests exactly the delivery
        the detector has been failing on.
        """
        tbl = self.knob_miss_table()
        if tbl.empty:
            return None, "no rounds scored yet; sampling the knob space uniformly"
        worst_per_dim = (
            tbl.sort_values("smoothed_miss_rate", ascending=False)
            .groupby("knob", sort=False)
            .first()
            .reset_index()
        )
        best = worst_per_dim.sort_values("smoothed_miss_rate", ascending=False).iloc[0]
        if float(best["smoothed_miss_rate"]) <= 0.0:
            return None, (
                "every variant so far was caught; no weak knob to exploit, "
                "sampling uniformly"
            )
        dim, value = str(best["knob"]), str(best["value"])
        rate, n = float(best["mean_miss_rate"]), int(best["rounds"])
        plural = "s" if n != 1 else ""
        return {dim: value}, (
            f"biasing toward {dim}={value} because misses cluster there "
            f"({rate:.0%} of sessions missed over {n} round{plural})"
        )

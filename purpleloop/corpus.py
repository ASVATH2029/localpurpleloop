"""Append-only, round-partitioned corpus.

Owner: Cloud Specialist (plan Phase 1).

One Parquet file per round under data/corpus/round_NNN.parquet. Rounds are
never rewritten, so the training set at round N is exactly the union of rounds
0..N and any earlier result stays reproducible.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import FEATURE_NAMES, META_COLUMNS


class Corpus:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.dir = self.root / "corpus"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- write
    def append_round(self, features: pd.DataFrame, round_no: int) -> Path:
        """Write one round's feature rows. Refuses to overwrite an existing round."""
        path = self.dir / f"round_{round_no:03d}.parquet"
        if path.exists():
            raise FileExistsError(
                f"round {round_no} already in the corpus ({path}); the corpus is append-only"
            )
        out = features.reindex(columns=FEATURE_NAMES + META_COLUMNS).copy()
        out["round"] = round_no
        out.to_parquet(path, index=False)
        return path

    def save_events(self, events: pd.DataFrame, round_no: int) -> Path:
        """Keep the raw events too, for inspection and the write-up."""
        path = self.events_dir / f"round_{round_no:03d}.parquet"
        events.to_parquet(path, index=False)
        return path

    # ----------------------------------------------------------------- read
    def rounds_present(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("round_*.parquet"))

    def load(self, max_round: int | None = None) -> pd.DataFrame:
        """Load rounds 0..max_round inclusive (all rounds if None)."""
        paths = sorted(self.dir.glob("round_*.parquet"))
        frames = []
        for p in paths:
            r = int(p.stem.split("_")[1])
            if max_round is None or r <= max_round:
                frames.append(pd.read_parquet(p))
        if not frames:
            return pd.DataFrame(columns=FEATURE_NAMES + META_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def known_variant_ids(self, max_round: int | None = None) -> set[str]:
        """Attack variant_ids already in the corpus (benign kinds excluded)."""
        df = self.load(max_round)
        if df.empty:
            return set()
        return set(df.loc[df["label"] == 1, "variant_id"].unique())

    def summary(self) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("round")
            .agg(sessions=("label", "size"), attacks=("label", "sum"))
            .assign(benign=lambda d: d["sessions"] - d["attacks"])
        )

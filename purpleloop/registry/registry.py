"""Versioned rule/threshold registry in SQLite.

Owner: ML Engineer #2 (plan Phase 3 / section 3 of the workaround plan).

Every round's candidate gets a row -- promoted or rejected, with the reason.
Nothing is ever overwritten, which buys three things:

  * governed self-updating: a candidate becomes active only by passing the gate
  * an audit trail: why each version was accepted or refused
  * rollback for free: flip which row carries promoted = 1
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DDL = """
CREATE TABLE IF NOT EXISTS rule_versions (
    version          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id          TEXT    NOT NULL,
    created_round    INTEGER NOT NULL,
    held_out_recall  REAL,
    benign_fpr       REAL,
    suite_recall     REAL,
    promoted         INTEGER NOT NULL DEFAULT 0,
    decision         TEXT    NOT NULL,
    reason           TEXT,
    candidate_kind   TEXT    NOT NULL DEFAULT 'retrained',
    model_path       TEXT,
    rule_body        TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS rollbacks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    from_version INTEGER,
    to_version   INTEGER,
    at_round     INTEGER,
    reason       TEXT,
    created_at   TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(DDL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------------- write
    def add_version(
        self,
        *,
        rule_id: str,
        created_round: int,
        held_out_recall: float | None,
        benign_fpr: float | None,
        suite_recall: float | None,
        decision: str,
        reason: str,
        rule_body: dict,
        model_path: str | None = None,
        candidate_kind: str = "retrained",
        promote: bool = False,
    ) -> int:
        """Insert a candidate version. promote=True also makes it active."""
        if decision not in {"promoted", "rejected"}:
            raise ValueError("decision must be 'promoted' or 'rejected'")
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO rule_versions
                   (rule_id, created_round, held_out_recall, benign_fpr, suite_recall,
                    promoted, decision, reason, candidate_kind, model_path, rule_body, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rule_id, created_round, held_out_recall, benign_fpr, suite_recall,
                    0, decision, reason, candidate_kind, model_path,
                    json.dumps(rule_body, sort_keys=True), _now(),
                ),
            )
            version = int(cur.lastrowid)
            if promote:
                c.execute("UPDATE rule_versions SET promoted = 0 WHERE promoted = 1")
                c.execute("UPDATE rule_versions SET promoted = 1 WHERE version = ?", (version,))
        return version

    def rollback(self, to_version: int, at_round: int, reason: str = "manual rollback") -> None:
        """Make a previous version active again. Nothing is deleted."""
        with self._conn() as c:
            row = c.execute(
                "SELECT version, decision FROM rule_versions WHERE version = ?", (to_version,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no such version: {to_version}")
            if row["decision"] != "promoted":
                raise ValueError(
                    f"version {to_version} was never promoted; it cannot be rolled back to"
                )
            cur = c.execute("SELECT version FROM rule_versions WHERE promoted = 1").fetchone()
            from_version = int(cur["version"]) if cur else None
            c.execute("UPDATE rule_versions SET promoted = 0 WHERE promoted = 1")
            c.execute("UPDATE rule_versions SET promoted = 1 WHERE version = ?", (to_version,))
            c.execute(
                """INSERT INTO rollbacks (from_version, to_version, at_round, reason, created_at)
                   VALUES (?,?,?,?,?)""",
                (from_version, to_version, at_round, reason, _now()),
            )

    # ----------------------------------------------------------------- read
    def get_active(self) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rule_versions WHERE promoted = 1 ORDER BY version DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["rule_body"] = json.loads(d["rule_body"])
        return d

    def history(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM rule_versions ORDER BY version", c)

    def rollback_log(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM rollbacks ORDER BY id", c)

    def audit_trail(self) -> list[str]:
        """Human-readable log lines, the kind a reviewer actually reads."""
        lines = []
        for _, r in self.history().iterrows():
            mark = "ACTIVE" if r["promoted"] else ("ok" if r["decision"] == "promoted" else "REJECTED")
            lines.append(
                f"v{int(r['version']):<3} round {int(r['created_round']):<3} "
                f"[{r['candidate_kind']}] {r['decision']:<9} {mark:<8} "
                f"held_out_recall={_fmt(r['held_out_recall'])} "
                f"suite_recall={_fmt(r['suite_recall'])} "
                f"benign_fpr={_fmt(r['benign_fpr'])} :: {r['reason']}"
            )
        for _, r in self.rollback_log().iterrows():
            lines.append(
                f"ROLLBACK at round {int(r['at_round'])}: "
                f"v{r['from_version']} -> v{int(r['to_version'])} :: {r['reason']}"
            )
        return lines


def _fmt(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "   n/a"
        return f"{float(x):6.3f}"
    except (TypeError, ValueError):
        return "   n/a"

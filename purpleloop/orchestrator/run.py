"""The autonomous round loop.

Owner: ML Engineer #2 (plan Phase 4).

One round, in order -- the order is the point:

  1. pick this round's brand-new attack variant, biased toward the knob
     dimension where the previous rounds' misses clustered
  2. generate it, plus fresh benign traffic for the same simulated day
  3. SCORE IT HELD OUT against the current active detector, before it is
     allowed anywhere near the training set
  4. append the round (attack + benign) to the append-only corpus
  5. retrain a candidate detector on the whole corpus
  6. recalibrate the threshold against a fresh benign sample and the FPR budget
  7. run the regression suite: re-rolled instances of every PREVIOUSLY seen
     variant, scored by both the candidate and the active version
  8. the gate promotes the candidate only if it regresses on neither axis;
     otherwise it is logged, rejected, and the old version stays live
  9. log the round, and attribute the misses to knob values for step 1 of the
     next round

Steps 3 and 4 must not be reordered: heldout_eval() asserts the variant is
absent from the corpus and raises if it is not.
"""
from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import schema
from ..corpus import Corpus
from ..features import extract_features
from ..gen import generate_benign, generate_variant, sample_knobs
from ..harness import heldout_eval
from ..model import calibrate_threshold, evaluate, train
from ..registry import Registry, evaluate_gate
from .policy import MissHistory

log = logging.getLogger("purpleloop")

ROUND_LOG_COLUMNS = [
    "round", "variant_id", "family",
    "delay_profile", "batch_size", "call_order", "identity_reuse",
    "biased_knob", "biased_value", "bias_reason",
    "held_out_recall", "n_heldout_sessions", "n_caught", "mean_heldout_score",
    "mean_anomaly",
    "active_threshold", "candidate_threshold",
    "benign_fpr", "active_fpr_fresh",
    "suite_recall_candidate", "suite_recall_active",
    "suite_fpr_candidate", "suite_fpr_active",
    "decision", "promoted", "version", "reason",
    "corpus_rows", "n_variants_known", "round_events", "elapsed_s",
]


def _seed(base: int, *parts) -> int:
    """Deterministic seed from a base plus string/int parts (hash-stable)."""
    h = int(base) & 0x7FFFFFFF
    for p in parts:
        h = (h * 1000003 + zlib.crc32(str(p).encode())) & 0x7FFFFFFF
    return h or 1


def setup_logging(log_dir: Path, verbose: bool = True) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_dir / "loop.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    log.propagate = False


class LoopRunner:
    def __init__(self, cfg: dict, root: str | Path):
        self.cfg = cfg
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.logs_dir = self.root / "logs"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.corpus = Corpus(self.data_dir)
        self.registry = Registry(self.data_dir / "registry.db")
        self.history = MissHistory()

        self.base_seed = int(cfg["run"]["seed"])
        self.rng = np.random.default_rng(self.base_seed)
        self.round_log_path = self.logs_dir / "rounds.csv"

    # ------------------------------------------------------------- helpers
    @property
    def c_corpus(self) -> dict:
        return self.cfg["corpus"]

    @property
    def c_det(self) -> dict:
        return self.cfg["detector"]

    @property
    def fpr_budget(self) -> float:
        return float(self.cfg["threshold"]["fpr_budget"])

    def _tolerances(self) -> dict:
        """Gate tolerances, from config."""
        keys = ("recall_tolerance", "fpr_tolerance", "budget_tolerance")
        return {k: float(v) for k, v in self.cfg["governance"].items() if k in keys}

    def _benign_features(self, n: int, tag: str, round_no: int) -> pd.DataFrame:
        """A fresh benign sample -- new seed every call, never reused."""
        ev = generate_benign(n, seed=_seed(self.base_seed, tag, round_no), round_no=round_no)
        return extract_features(ev)

    def _save_detector(self, det, name: str) -> Path:
        path = self.models_dir / f"{name}.joblib"
        joblib.dump(det, path)
        return path

    def _active(self):
        """(registry row, detector, threshold) for the live version, or Nones."""
        row = self.registry.get_active()
        if row is None:
            return None, None, None
        det = joblib.load(row["model_path"])
        return row, det, float(row["rule_body"]["threshold"])

    def _regression_suite(self, variant_ids, round_no: int) -> pd.DataFrame:
        """Re-roll every previously seen variant with fresh seeds, plus fresh benign.

        Both the candidate and the active version have trained on these
        variants, so the comparison is symmetric: it asks whether the candidate
        still catches what the loop already knew, not whether it memorised a
        specific sample.
        """
        frames = []
        n_inst = int(self.c_corpus["suite_instances_per_variant"])
        for vid in sorted(variant_ids):
            fam, knobs = schema.parse_variant_id(vid)
            if fam not in schema.FAMILIES:
                continue
            frames.append(generate_variant(
                fam, knobs,
                seed=_seed(self.base_seed, "suite", round_no, vid),
                n_sessions=n_inst,
                round_no=round_no,
            ))
        suite_benign = generate_benign(
            int(self.c_corpus["suite_benign_sessions"]),
            seed=_seed(self.base_seed, "suite-benign", round_no),
            round_no=round_no,
        )
        frames.append(suite_benign)
        return extract_features(pd.concat(frames, ignore_index=True))

    def _cap_round_events(self, atk: pd.DataFrame, ben: pd.DataFrame) -> pd.DataFrame:
        """Keep each round's event count inside the configured bound.

        Attack sessions are always kept; benign sessions are dropped whole,
        never truncated mid-session, so no partial episode enters the corpus.
        """
        cap = int(self.cfg["run"]["max_events_per_round"])
        total = len(atk) + len(ben)
        if total <= cap:
            return pd.concat([atk, ben], ignore_index=True)

        budget = max(0, cap - len(atk))
        sizes = ben.groupby("session_id", sort=False).size()
        keep, used = [], 0
        for sid, size in sizes.items():
            if used + size > budget:
                continue
            keep.append(sid)
            used += size
        log.warning(
            "round event cap %d hit (%d events): keeping %d/%d benign sessions",
            cap, total, len(keep), len(sizes),
        )
        return pd.concat(
            [atk, ben[ben["session_id"].isin(keep)]], ignore_index=True
        )

    def _append_round_log(self, row: dict) -> None:
        df = pd.DataFrame([row]).reindex(columns=ROUND_LOG_COLUMNS)
        header = not self.round_log_path.exists()
        df.to_csv(self.round_log_path, mode="a", header=header, index=False)

    def _rule_body(self, det, threshold: float, cal: dict, extra: dict) -> dict:
        """The versioned artifact: what the active decision rule actually is."""
        body = {
            "model_kind": det.model_kind,
            "threshold": float(threshold),
            "fpr_budget": self.fpr_budget,
            "n_features": len(det.feature_names),
            "feature_set_hash": zlib.crc32(",".join(det.feature_names).encode()),
            "n_train_rows": int(det.n_train),
            "trained_through_round": int(det.trained_rounds),
            "calibration": {k: v for k, v in cal.items() if k != "threshold"},
            "detector_params": dict(self.c_det),
        }
        body.update(extra)
        return body

    # ----------------------------------------------------------- bootstrap
    def bootstrap(self) -> None:
        """Round 0: the hand-authored seed variants become version 1.

        No held-out number is recorded here -- there is no prior model to hold
        anything out from. The curve starts at round 1.
        """
        if 0 in self.corpus.rounds_present():
            log.info("round 0 already present; skipping bootstrap")
            return

        log.info("=" * 78)
        log.info("ROUND 0  bootstrap: hand-authored seed variants")
        n_sessions = max(2, int(self.c_corpus["attack_sessions_per_round"]) // 2)
        frames = []
        for fam, variants in schema.SEED_VARIANTS.items():
            for i, knobs in enumerate(variants):
                frames.append(generate_variant(
                    fam, knobs,
                    seed=_seed(self.base_seed, "seed", fam, i),
                    n_sessions=n_sessions,
                    round_no=0,
                ))
        atk = pd.concat(frames, ignore_index=True)
        ben = generate_benign(
            int(self.c_corpus["benign_sessions_per_round"]) * 2,
            seed=_seed(self.base_seed, "seed-benign"),
            round_no=0,
        )
        events = self._cap_round_events(atk, ben)
        feats = extract_features(events)
        self.corpus.save_events(events, 0)
        self.corpus.append_round(feats, 0)
        log.info(
            "round 0 corpus: %d sessions (%d attack across %d seed variants, %d benign)",
            len(feats), int(feats.label.sum()),
            feats.loc[feats.label == 1, "variant_id"].nunique(),
            int((feats.label == 0).sum()),
        )

        det = train(self.corpus.load(), self.c_det, seed=_seed(self.base_seed, "fit", 0))
        cal_benign = self._benign_features(
            int(self.c_corpus["calibration_benign_sessions"]), "cal", 0
        )
        cal = calibrate_threshold(det, cal_benign, self.fpr_budget)

        suite = self._regression_suite(self.corpus.known_variant_ids(), round_no=0)
        cand_eval = evaluate(det, cal["threshold"], suite)
        gate = evaluate_gate(
            candidate_suite_recall=cand_eval["recall"],
            candidate_calibrated_fpr=cal["benign_fpr"],
            candidate_suite_fpr=cand_eval["fpr"],
            active_suite_recall=None,
            active_suite_fpr=None,
            fpr_budget=self.fpr_budget,
            **self._tolerances(),
        )
        path = self._save_detector(det, "round000_retrained")
        version = self.registry.add_version(
            rule_id="purple-loop-detector",
            created_round=0,
            held_out_recall=None,
            benign_fpr=cal["benign_fpr"],
            suite_recall=cand_eval["recall"],
            decision=gate.decision,
            reason=gate.reason,
            rule_body=self._rule_body(det, cal["threshold"], cal, {"bootstrap": True}),
            model_path=str(path),
            candidate_kind="bootstrap",
            promote=gate.promote,
        )
        log.info(
            "round 0 -> v%d %s | threshold %.4f | benign FPR %.4f | suite recall %.3f",
            version, gate.decision, cal["threshold"], cal["benign_fpr"], cand_eval["recall"],
        )
        log.info("top features: %s", ", ".join(det.feature_importance(6).index))
        self._append_round_log({
            "round": 0, "variant_id": "(bootstrap)", "family": "(seeds)",
            "biased_knob": "", "bias_reason": "bootstrap round",
            "candidate_threshold": cal["threshold"], "benign_fpr": cal["benign_fpr"],
            "suite_recall_candidate": cand_eval["recall"],
            "suite_fpr_candidate": cand_eval["fpr"],
            "decision": gate.decision, "promoted": int(gate.promote),
            "version": version, "reason": gate.reason,
            "corpus_rows": len(feats),
            "n_variants_known": len(self.corpus.known_variant_ids()),
            "round_events": len(events), "elapsed_s": 0.0,
        })

    # ---------------------------------------------------------------- round
    def run_round(self, r: int) -> dict:
        t0 = time.time()
        log.info("=" * 78)

        corpus_before = self.corpus.load()
        known = set(corpus_before.loc[corpus_before["label"] == 1, "variant_id"].unique())

        # --- 1. choose this round's variant, biased toward the weak knob -----
        if r <= int(self.cfg["policy"]["seed_rounds"]):
            bias, bias_reason = None, (
                f"round {r} <= seed_rounds; exploring the knob space uniformly"
            )
        else:
            bias, bias_reason = self.history.choose_bias()
        bias_dim = next(iter(bias)) if bias else ""
        bias_val = bias[bias_dim] if bias else ""
        family, knobs = sample_knobs(
            self.rng,
            bias=bias,
            epsilon=float(self.cfg["policy"]["epsilon"]),
            exclude=known,
        )
        vid = schema.variant_id(family, knobs)
        log.info("ROUND %d  %s", r, family)
        log.info("  policy: %s", bias_reason)
        log.info("  variant: %s", ",".join(f"{k}={knobs[k]}" for k in schema.KNOB_NAMES))

        # --- 2. generate the variant and this round's benign traffic ---------
        atk_events = generate_variant(
            family, knobs,
            seed=_seed(self.base_seed, "atk", r, vid),
            n_sessions=int(self.c_corpus["attack_sessions_per_round"]),
            round_no=r,
        )
        ben_events = generate_benign(
            int(self.c_corpus["benign_sessions_per_round"]),
            seed=_seed(self.base_seed, "ben", r),
            round_no=r,
        )
        events = self._cap_round_events(atk_events, ben_events)
        feats = extract_features(events)
        atk_feats = feats[feats["label"] == 1]
        ben_feats = feats[feats["label"] == 0]

        # --- 3. HELD-OUT EVAL, before the corpus ever sees this variant ------
        active_row, active_det, active_thr = self._active()
        if active_det is None:
            raise RuntimeError("no active detector; run bootstrap() first")
        ho = heldout_eval(
            active_det, active_thr, atk_feats,
            variant_id=vid, corpus=corpus_before,
        )
        active_fresh = evaluate(active_det, active_thr, ben_feats)
        log.info(
            "  HELD OUT: caught %d/%d sessions -> recall %.3f (mean score %.3f, thr %.3f)",
            ho["n_caught"], ho["n_sessions"], ho["held_out_recall"],
            ho["mean_score"], active_thr,
        )
        log.info("  active version FPR on this round's fresh benign: %.4f", active_fresh["fpr"])

        # --- 4. append to the append-only corpus -----------------------------
        self.corpus.save_events(events, r)
        self.corpus.append_round(feats, r)
        corpus_after = self.corpus.load()

        # --- 5/6. retrain and recalibrate ------------------------------------
        cand = train(corpus_after, self.c_det, seed=_seed(self.base_seed, "fit", r))
        cal_benign = self._benign_features(
            int(self.c_corpus["calibration_benign_sessions"]), "cal", r
        )
        cal = calibrate_threshold(cand, cal_benign, self.fpr_budget)

        # --- 7. regression suite over PREVIOUSLY seen variants ---------------
        suite = self._regression_suite(known, round_no=r)
        cand_eval = evaluate(cand, cal["threshold"], suite)
        act_eval = evaluate(active_det, active_thr, suite)
        log.info(
            "  suite (%d variants): recall active %.3f -> candidate %.3f | "
            "benign FPR active %.4f -> candidate %.4f",
            len(known), act_eval["recall"], cand_eval["recall"],
            act_eval["fpr"], cand_eval["fpr"],
        )

        # --- 8. the gate ------------------------------------------------------
        gate = evaluate_gate(
            candidate_suite_recall=cand_eval["recall"],
            candidate_calibrated_fpr=cal["benign_fpr"],
            candidate_suite_fpr=cand_eval["fpr"],
            active_suite_recall=act_eval["recall"],
            active_suite_fpr=act_eval["fpr"],
            fpr_budget=self.fpr_budget,
            **self._tolerances(),
        )
        path = self._save_detector(cand, f"round{r:03d}_retrained")
        version = self.registry.add_version(
            rule_id="purple-loop-detector",
            created_round=r,
            held_out_recall=ho["held_out_recall"],
            benign_fpr=cal["benign_fpr"],
            suite_recall=cand_eval["recall"],
            decision=gate.decision,
            reason=gate.reason,
            rule_body=self._rule_body(cand, cal["threshold"], cal, {
                "new_variant": vid,
                "held_out_recall_at_promotion": ho["held_out_recall"],
            }),
            model_path=str(path),
            candidate_kind="retrained",
            promote=gate.promote,
        )
        log.info("  gate: %s v%d -- %s", gate.decision.upper(), version, gate.reason)

        # --- 8b. governance test: a deliberately over-aggressive threshold ---
        inject = self.cfg["governance"].get("inject_aggressive_candidate_round")
        if inject is not None and int(inject) == r:
            self._propose_aggressive(r, cand, suite, cand_eval, act_eval, active_row)

        # --- 9. attribute misses, for next round's policy --------------------
        self.history.add(
            round_no=r, family=family, knobs=knobs,
            miss_rate=1.0 - ho["held_out_recall"],
        )

        elapsed = time.time() - t0
        row = {
            "round": r, "variant_id": vid, "family": family,
            **{k: knobs[k] for k in schema.KNOB_NAMES},
            "biased_knob": bias_dim, "biased_value": bias_val,
            "bias_reason": bias_reason,
            "held_out_recall": ho["held_out_recall"],
            "n_heldout_sessions": ho["n_sessions"], "n_caught": ho["n_caught"],
            "mean_heldout_score": ho["mean_score"], "mean_anomaly": ho["mean_anomaly"],
            "active_threshold": active_thr, "candidate_threshold": cal["threshold"],
            "benign_fpr": cal["benign_fpr"], "active_fpr_fresh": active_fresh["fpr"],
            "suite_recall_candidate": cand_eval["recall"],
            "suite_recall_active": act_eval["recall"],
            "suite_fpr_candidate": cand_eval["fpr"],
            "suite_fpr_active": act_eval["fpr"],
            "decision": gate.decision, "promoted": int(gate.promote),
            "version": version, "reason": gate.reason,
            "corpus_rows": len(corpus_after),
            "n_variants_known": len(known) + 1,
            "round_events": len(events), "elapsed_s": round(elapsed, 2),
        }
        self._append_round_log(row)
        return row

    def _propose_aggressive(self, r, cand, suite, cand_eval, act_eval, active_row) -> None:
        """Propose a threshold tuned for recall at the cost of the FPR budget.

        This is the Cybersecurity Analyst's deliberate governance test (plan
        Phase 3): a candidate that would catch more attacks but breach the
        false-positive budget. The gate must refuse it. Labelled as its own
        candidate_kind so nobody mistakes it for an organic result.
        """
        mult = float(self.cfg["governance"]["aggressive_fpr_multiplier"])
        loose_budget = self.fpr_budget * mult
        cal_benign = self._benign_features(
            int(self.c_corpus["calibration_benign_sessions"]), "cal-aggressive", r
        )
        loose = calibrate_threshold(cand, cal_benign, loose_budget)
        loose_eval = evaluate(cand, loose["threshold"], suite)

        gate = evaluate_gate(
            candidate_suite_recall=loose_eval["recall"],
            candidate_calibrated_fpr=loose["benign_fpr"],
            candidate_suite_fpr=loose_eval["fpr"],
            active_suite_recall=act_eval["recall"],
            active_suite_fpr=act_eval["fpr"],
            fpr_budget=self.fpr_budget,
            **self._tolerances(),
        )
        version = self.registry.add_version(
            rule_id="purple-loop-detector",
            created_round=r,
            held_out_recall=None,
            benign_fpr=loose["benign_fpr"],
            suite_recall=loose_eval["recall"],
            decision=gate.decision,
            reason=f"governance test ({mult:g}x FPR budget): {gate.reason}",
            rule_body=self._rule_body(cand, loose["threshold"], loose, {
                "aggressive_budget": loose_budget,
                "note": "deliberately over-permissive threshold, proposed to exercise the gate",
            }),
            model_path=None,
            candidate_kind="aggressive_threshold",
            promote=gate.promote,
        )
        log.info(
            "  governance test v%d: threshold %.4f would reach suite recall %.3f "
            "at benign FPR %.4f -> %s",
            version, loose["threshold"], loose_eval["recall"],
            loose["benign_fpr"], gate.decision.upper(),
        )
        if gate.promote:
            log.error(
                "  governance test was PROMOTED -- the gate is not doing its job; "
                "check governance.fpr_tolerance"
            )

    # ------------------------------------------------------------------ run
    def run(self, rounds: int | None = None) -> pd.DataFrame:
        rounds = int(rounds or self.cfg["run"]["rounds"])
        stop_file = self.root / "STOP"
        self.bootstrap()

        done = [x for x in self.corpus.rounds_present() if x > 0]
        start = (max(done) + 1) if done else 1

        for r in range(start, rounds + 1):
            if stop_file.exists():
                log.warning("kill switch %s present; stopping after round %d", stop_file, r - 1)
                break
            self.run_round(r)
            nap = float(self.cfg["run"].get("sleep_between_rounds_s", 0) or 0)
            if nap:
                time.sleep(nap)

        log.info("=" * 78)
        log.info("loop finished; %d rounds in the corpus", len(self.corpus.rounds_present()) - 1)
        for line in self.registry.audit_trail():
            log.info("  %s", line)
        return self.round_log()

    def round_log(self) -> pd.DataFrame:
        if not self.round_log_path.exists():
            return pd.DataFrame(columns=ROUND_LOG_COLUMNS)
        return pd.read_csv(self.round_log_path)

    def save_policy_tables(self) -> None:
        """Write the knob-level miss table used by the policy, for the write-up."""
        tbl = self.history.knob_miss_table()
        if not tbl.empty:
            tbl.to_csv(self.logs_dir / "knob_miss_rates.csv", index=False)
        hist = self.history.table()
        if not hist.empty:
            hist.to_csv(self.logs_dir / "miss_history.csv", index=False)
        trail = self.registry.audit_trail()
        (self.logs_dir / "registry_audit.txt").write_text(
            "\n".join(trail) + "\n", encoding="utf-8"
        )
        self.registry.history().to_csv(self.logs_dir / "registry_history.csv", index=False)
        json.dump(
            {"rounds_present": self.corpus.rounds_present()},
            open(self.logs_dir / "corpus_rounds.json", "w"),
            indent=2,
        )

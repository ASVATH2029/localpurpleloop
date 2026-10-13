"""Command line entry point.

    python -m purpleloop run        --rounds 20
    python -m purpleloop report
    python -m purpleloop registry
    python -m purpleloop rollback   --to-version 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .orchestrator.run import LoopRunner, setup_logging

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _runner(args) -> LoopRunner:
    cfg = load_config(args.config)
    if getattr(args, "rounds", None):
        cfg["run"]["rounds"] = int(args.rounds)
    if getattr(args, "seed", None):
        cfg["run"]["seed"] = int(args.seed)
    root = Path(args.root or ROOT)
    setup_logging(root / "logs", verbose=not getattr(args, "quiet", False))
    return LoopRunner(cfg, root)


def cmd_run(args) -> int:
    runner = _runner(args)
    runner.run(rounds=args.rounds)
    runner.save_policy_tables()
    cmd_report(args, runner=runner)
    return 0


def cmd_report(args, runner: LoopRunner | None = None) -> int:
    from . import plots

    runner = runner or _runner(args)
    log = runner.round_log()
    scored = log[log["round"] > 0]
    if scored.empty:
        print("no scored rounds yet; run the loop first")
        return 1

    figs = runner.root / "docs" / "figures"
    plots.generalisation_curve(log, figs / "generalisation_curve.png")
    plots.suite_recall_chart(log, figs / "suite_recall.png")
    knobs = runner.history.knob_miss_table()
    if knobs.empty:
        kp = runner.logs_dir / "knob_miss_rates.csv"
        if kp.exists():
            import pandas as pd
            knobs = pd.read_csv(kp)
    if not knobs.empty:
        plots.knob_miss_chart(knobs, figs / "knob_miss_rates.png")

    first, last = scored.head(5), scored.tail(5)
    print()
    print("=" * 72)
    print("PURPLE LOOP RESULTS")
    print("=" * 72)
    print(f"rounds scored held-out        : {len(scored)}")
    print(f"mean held-out recall, first 5 : {first['held_out_recall'].mean():.3f}")
    print(f"mean held-out recall, last 5  : {last['held_out_recall'].mean():.3f}")
    print(f"benign FPR  mean / max        : {scored['benign_fpr'].mean():.4f}"
          f" / {scored['benign_fpr'].max():.4f}")
    print(f"candidates promoted / rejected: {int(scored['promoted'].sum())}"
          f" / {int((scored['promoted'] == 0).sum())}")
    print(f"figures                       : {figs}")
    print()
    cols = ["round", "family", "delay_profile", "call_order", "held_out_recall",
            "benign_fpr", "decision"]
    print(scored[cols].to_string(index=False))
    print()
    print("registry audit trail")
    print("-" * 72)
    for line in runner.registry.audit_trail():
        print(" ", line)
    return 0


def cmd_ablate(args) -> int:
    """Run the loop once per feature group and report whether the claim holds."""
    from . import ablation, plots

    runner = _runner(args)
    results = ablation.run_ablation(runner.cfg, runner.root, rounds=int(args.rounds))
    summary = ablation.summarise(results)
    out_csv = runner.logs_dir / "ablation_summary.csv"
    summary.to_csv(out_csv, index=False)
    fig = plots.ablation_chart(results, runner.root / "docs" / "figures" / "ablation.png")

    print()
    print("=" * 72)
    print("ABLATION: WHICH FEATURES CARRY THE GENERALISATION")
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(ablation.verdict(summary))
    print()
    print("summary:", out_csv)
    print("figure :", fig)
    return 0


def cmd_reset(args) -> int:
    """Delete generated artifacts so the loop can be re-run from round 0."""
    import shutil

    root = Path(args.root or ROOT)
    targets = [root / "data", root / "logs", root / "ablation",
               root / "docs" / "figures"]
    if not args.yes:
        print("this will delete:")
        for t in targets:
            print("  ", t)
        print("re-run with --yes to confirm")
        return 1
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            print("removed", t)
    return 0


def cmd_registry(args) -> int:
    runner = _runner(args)
    for line in runner.registry.audit_trail():
        print(line)
    active = runner.registry.get_active()
    if active:
        print(f"\nACTIVE: v{active['version']} from round {active['created_round']} "
              f"threshold={active['rule_body']['threshold']:.4f}")
    return 0


def cmd_rollback(args) -> int:
    runner = _runner(args)
    runner.registry.rollback(
        to_version=int(args.to_version),
        at_round=int(args.at_round or 999),
        reason=args.reason,
    )
    print(f"rolled back to v{args.to_version}")
    for line in runner.registry.audit_trail():
        print(line)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="purpleloop", description=__doc__)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--root", default=None, help="project root (data/ and logs/ live here)")
    p.add_argument("-q", "--quiet", action="store_true", help="no console logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the autonomous loop")
    pr.add_argument("--rounds", type=int, default=None)
    pr.add_argument("--seed", type=int, default=None)
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="rebuild figures and print the summary")
    prep.set_defaults(func=cmd_report)

    pg = sub.add_parser("registry", help="print the rule registry audit trail")
    pg.set_defaults(func=cmd_registry)

    pa = sub.add_parser("ablate", help="compare feature groups (delivery vs technique)")
    pa.add_argument("--rounds", type=int, default=12)
    pa.set_defaults(func=cmd_ablate)

    pz = sub.add_parser("reset", help="delete data/, logs/ and figures to re-run from scratch")
    pz.add_argument("--yes", action="store_true", help="confirm the deletion")
    pz.set_defaults(func=cmd_reset)

    pb = sub.add_parser("rollback", help="make a previous rule version active again")
    pb.add_argument("--to-version", required=True)
    pb.add_argument("--at-round", default=None)
    pb.add_argument("--reason", default="manual rollback demo")
    pb.set_defaults(func=cmd_rollback)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

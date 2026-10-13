# Purple Loop

An autonomous, **governed** self-training detection loop. It attacks itself,
retrains on everything it has seen, and is scored on whether it catches
**brand-new attack variants it was never trained on**.

Runs entirely locally. No AWS account, no cloud credentials, no network calls —
the events are synthetic but shaped like CloudTrail records.

```powershell
.\run.ps1 setup      # create .venv, install requirements
.\run.ps1 test       # 60 tests, including the anti-leakage guards
.\run.ps1 run 20     # the loop: 20 rounds, ~100 seconds
.\run.ps1 report     # figures + summary
```

On Linux/macOS/Git Bash use `make venv && make test && make run`.
For a hard sandbox: `docker build -t purple-loop . && docker run --rm --network none purple-loop`.

---

## What one round does

The order matters, and it is enforced in code:

| # | Step | Where |
|---|------|-------|
| 1 | Pick a brand-new attack variant, **biased toward the knob value whose misses cluster** | [policy.py](purpleloop/orchestrator/policy.py) |
| 2 | Generate it, plus fresh benign traffic for the same simulated day | [gen/](purpleloop/gen/) |
| 3 | **Score it held out** against the current detector — before training on it | [harness/heldout.py](purpleloop/harness/heldout.py) |
| 4 | Append the round to the append-only corpus | [corpus.py](purpleloop/corpus.py) |
| 5 | Retrain a candidate on the whole corpus | [model/detector.py](purpleloop/model/detector.py) |
| 6 | Recalibrate the threshold to the benign FPR budget | [model/calibrate.py](purpleloop/model/calibrate.py) |
| 7 | Run the regression suite over previously seen variants | [run.py](purpleloop/orchestrator/run.py) |
| 8 | **The gate** promotes only if neither axis regressed; else logs a reject | [registry/gate.py](purpleloop/registry/gate.py) |
| 9 | Attribute the misses to knob values, for step 1 next round | [policy.py](purpleloop/orchestrator/policy.py) |

Steps 3 and 4 cannot be swapped: `heldout_eval()` asserts the variant is absent
from the corpus and **raises** rather than reporting an inflated number.

## The attack space

Three families — IAM escalation, S3 exfiltration, CloudTrail tampering — each
mutated along four knobs: `delay_profile` (burst → slow → jittered),
`batch_size`, `call_order` (canonical → reversed → interleaved → split) and
`identity_reuse`. That is 540 distinct variants; a run visits 20 of them.

Benign traffic includes deliberate **hard negatives**, which is what makes the
1% false-positive budget mean anything:

| Benign kind | Looks like | Differs by |
|---|---|---|
| `user_onboarding` | IAM escalation | grants ReadOnlyAccess, not AdministratorAccess |
| `backup_job` | S3 exfiltration | 40–160 reads, but on a non-sensitive bucket |
| `analyst_sensitive_query` | S3 exfiltration | sensitive bucket, but a handful of objects |
| `log_maintenance` | trail tampering | broadens audit scope instead of narrowing it |

## Results from the committed run

20 rounds, seed `20251013`, ~100 s total, 70k events, 3236 sessions in the
final corpus.

| Check | Result |
|---|---|
| Held-out recall, first 5 rounds → last 5 | **0.800 → 1.000** |
| Benign FPR, calibrated | **0.0100 every round** (budget 0.01) |
| Active version's FPR on fresh benign | 0.0118 mean |
| Regression-suite recall | 1.000 mean — nothing forgotten |
| Adaptive policy | biased 18/20 rounds, rotating `delay_profile=slow` (10), `call_order=split` (4), `batch_size=medium` (4) |
| Governed rejection | 1 candidate auto-rejected at 0.100 FPR vs the 0.01 budget |
| Rollback | demonstrated, v22 → v5 |

Figures land in `docs/figures/`. Full write-up, including what did **not**
work: **[docs/RESULTS.md](docs/RESULTS.md)**.

### Read this before quoting the curve

The curve rises, but it **saturates after one round**: round 1's
slow-timing variant was missed completely, and all 19 later variants were
caught outright. The rise is driven by a single early miss, not a gradual
climb.

An ablation that restricts the detector to delivery-only features
(`.\run.ps1 ablate`) reaches 0.896 mean held-out recall against 0.906 for the
full feature set — so the appealing claim that *technique* features are what
generalise is **not supported** by this setup. The synthetic attack and benign
distributions are separable from gross session shape alone.

What the run does establish: the loop is autonomous, the held-out measurement
is honest, false positives stayed inside budget while coverage grew, and the
self-updating is governed rather than blind. What it does not establish: that
generalisation requires the technique features. [docs/RESULTS.md](docs/RESULTS.md)
sets out the four concrete changes that would make the task hard enough to
answer that.

## Layout

```
config.yaml              every tunable; no number is hand-picked in code
purpleloop/
  schema.py              frozen contracts: events, API vocab, knobs, families
  gen/                   attacks.py (3 families + mutators), benign.py
  features/extract.py    session -> 67 features, split into delivery/technique
  model/                 detector.py (train/score), calibrate.py (threshold)
  harness/heldout.py     score-before-train, with the leakage assertion
  registry/              registry.py (SQLite versions), gate.py (promotion rule)
  orchestrator/          run.py (the loop), policy.py (weak-knob bias)
  ablation.py            delivery vs technique comparison
  plots.py               the figures
tests/                   60 tests
docs/RESULTS.md          findings, honestly, including the negative one
docs/TEAM-PLAN.md        phase-by-phase plan and who owns what
```

## Governance

Every candidate gets a row in `data/registry.db` — promoted or rejected, with
the reason. Nothing is overwritten, so rollback is just flipping which row is
active, and the audit trail reads like this:

```
v8   round 7  [retrained]            promoted  :: no regression on either axis
v9   round 7  [aggressive_threshold] REJECTED  :: calibrated benign FPR 0.1000
                                                  breaches the 0.0100 budget
```

`purpleloop registry` prints the trail; `purpleloop rollback --to-version N`
reverts.

## Reproducing

Everything is seeded from `config.yaml` (`run.seed`). The corpus is
append-only and partitioned by round, so a run can be stopped (`touch STOP`)
and resumed, and any round's state can be rebuilt. `data/`, `logs/` and
`ablation/` are gitignored — a 20-round run writes ~80 MB of trained forests.

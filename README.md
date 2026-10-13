# logalpurpleloop — Autonomous Governed Self-Training Detection Loop

> A detection system that attacks itself, learns from what it misses, and is governed by rules that prevent it from getting worse while doing it.

Runs **entirely locally**. No AWS account. No credentials. No network calls. The events are synthetic but shaped exactly like real CloudTrail records.

---

## What problem does this solve?

A traditional security detector is written once by a human who imagines a set of attack patterns. The moment an attacker changes the *timing*, *order*, or *identity* of their calls — not the technique, just the delivery — that detector misses them. The rule doesn't know what it doesn't know.

**Purple Loop fixes this by making the system its own red team.** Every round, it generates a brand-new variation of a known attack style it has never trained on, tests whether the current detector catches it *before* adding it to training, retrains on everything it has accumulated, and only promotes the new version if it provably doesn't get worse on anything it already knew. After 20 rounds it has stress-tested itself against 20 distinct attack variations across three families — and every model upgrade is logged, auditable, and reversible.

The "purple" is security terminology: red team attacks, blue team defends, purple team does both in a closed loop.

---

## Quick start

```powershell
cd E:\Projects\purple-loop

# 1. Set up the environment (first time only)
.\run.ps1 setup

# 2. Run all 60 tests — including the anti-cheat guards
.\run.ps1 test

# 3. Run the autonomous loop for 20 rounds (~100 seconds)
.\run.ps1 run 20

# 4. See figures + per-round summary
.\run.ps1 report

# 5. See the full governance audit trail
.\run.ps1 registry
```

On Linux/macOS/Git Bash: `make venv && make test && make run`

Hard sandbox (zero network): `docker build -t purple-loop . && docker run --rm --network none purple-loop`

---

## What one round does — the 9 steps

The order is not a convention. **Steps 3 and 4 cannot be swapped** — the harness enforces this in code and crashes if you try to cheat.

| # | Step | What happens |
|---|------|-------------|
| **1** | **Pick a variant** | The policy looks at past misses. Which attack style does the current detector struggle with? Next round targets that. First 2 rounds explore freely. |
| **2** | **Generate events** | Synthetic CloudTrail-shaped sessions for that variant, plus fresh benign traffic for the same simulated day. |
| **3** | **Score held-out** ⚠️ | Run the **current, unmodified** detector against this variant. The variant has never been in training. This is the honest measurement. |
| **4** | **Append to corpus** | Only now does the variant enter the training set. The corpus is append-only — no round can ever be edited or deleted. |
| **5** | **Retrain** | A fresh RandomForest is trained on the **entire** corpus — all rounds, all families, growing each round. |
| **6** | **Recalibrate threshold** | Set the alert cutoff at the highest value that keeps benign FPR ≤ 1%. Recalculated fresh every round. |
| **7** | **Regression suite** | Run the candidate against **all previously seen variants**. Did we forget anything we used to catch? |
| **8** | **Gate** 🔒 | Promote the candidate only if recall didn't drop AND false positives didn't rise — compared on one identical fresh sample. Otherwise log a reject and keep the old model. |
| **9** | **Update policy** | Record which attack styles were missed. Update smoothed miss-rate estimates. Round N+1 targets the weakest value. |

---

## The three attack families

All events are synthetic — shaped like real CloudTrail records but generated in-process. No real AWS account, no credentials, no API calls.

### 1 — IAM Escalation
**What the attacker does:** `CreateUser` → `AttachUserPolicy(AdministratorAccess)` → optionally `CreateAccessKey`

**Goal:** Create a persistent backdoor with full AWS control, disguised as routine account management.

**Why detection is hard:** Legitimate onboarding does the same IAM calls — it just grants `ReadOnlyAccess`, not `AdministratorAccess`. The detector cannot simply flag "IAM calls happened."

---

### 2 — S3 Exfiltration
**What the attacker does:** A burst of `GetObject` calls on sensitive buckets — `finance-reports`, `hr-data`, `customer-pii`.

**Goal:** Steal bulk data before the alert fires.

**Why detection is hard:** Backup jobs do high-volume reads too (just on non-sensitive buckets). Analysts legitimately query sensitive buckets (just a handful of objects). Neither alone is an attack.

---

### 3 — CloudTrail Tampering
**What the attacker does:** `StopLogging`, `DeleteTrail`, or `PutEventSelectors` with a narrowed scope.

**Goal:** Cover tracks so subsequent activity produces no audit trail.

**Why detection is hard:** `PutEventSelectors` is also called by defenders doing log maintenance — except they *broaden* scope instead of narrowing it. Same API call, opposite intent.

---

## The four mutation knobs — how the attacker evades detection

The technique (which APIs, which buckets) stays the same. What changes each round is the *delivery*:

| Knob | Values | What the attacker is doing |
|------|--------|---------------------------|
| `delay_profile` | burst / fast / spaced / slow / jittered | Slowing down or randomising timing so the session doesn't look automated |
| `batch_size` | small / medium / large | Breaking the attack into fewer or more API calls per batch |
| `call_order` | canonical / reversed / interleaved / split | Mixing attack calls with innocent filler, or doing them out of expected sequence |
| `identity_reuse` | single / two_identities / per_call_identity | Spreading calls across AWS identities so no single one looks suspicious |

**540 distinct variants** (3 families × 180 knob combinations). A 20-round run visits 20 of them plus 6 hand-authored seeds = 26 total.

---

## Hard negatives — why the 1% FPR budget is non-trivial

If benign traffic looks nothing like attacks, any detector trivially achieves 0% false positives. These four benign types are deliberately designed to overlap with attacks along one axis:

| Benign kind | Looks like | Differs by |
|------------|------------|------------|
| `user_onboarding` | IAM escalation — same `CreateUser` + `AttachUserPolicy` calls | Grants `ReadOnlyAccess`, not `AdministratorAccess` |
| `backup_job` | S3 exfiltration — high-volume `GetObject` | Reads a non-sensitive bucket |
| `analyst_sensitive_query` | S3 exfiltration — touches sensitive bucket | Only a handful of objects, not bulk |
| `log_maintenance` | CloudTrail tampering — same `PutEventSelectors` call | Broadens audit scope instead of narrowing it |

Without these, the FPR budget is free and means nothing.

---

## The 67 features — what the detector actually sees

The detector never sees raw event names or the variant ID. It sees only session-level statistics computed by `purpleloop/features/extract.py`:

**Delivery features (32)** — *how* the session was conducted:
- Inter-arrival timing: mean, std, max, median, entropy, burst ratio
- Session duration and log-duration
- Call volume: total events, unique API calls, resource concentration (Gini)
- Time-of-day slots, ordering bigrams (which API followed which)

**Technique features (35)** — *what* was done:
- Admin grants: `n_admin_grant`, `admin_grant_present`, `n_privilege_grant`
- Identity creation: `n_identity_create`
- Log coverage: `n_logging_disable`, `n_log_narrow`, `n_log_broaden`
- Sensitive data: `n_sensitive_reads`, `sensitive_read_frac`, `sensitive_bucket_touched`, `max_reads_one_sensitive_bucket`
- Per-category fractions, log-counts, and co-occurrence flags

---

## The adaptive policy — how the loop targets its own blind spots

After each round, `purpleloop/orchestrator/policy.py` looks at the sessions that were missed and records which knob values appeared in those misses. It computes a **smoothed miss rate** per knob value:

```
smoothed_miss_rate = (total_misses + PRIOR_WEIGHT × global_mean) / (n_rounds + PRIOR_WEIGHT)
```

`PRIOR_WEIGHT = 2.0` prevents one noisy round from dominating. The knob value with the highest smoothed miss rate wins. Next round's generator is told: "make a variant with `delay_profile=slow`" — not just "bias toward the timing dimension", but the specific failing value.

**Why this matters:** If you bias a *dimension* but sample a different value within it, you never gather evidence about the actual failure case — the estimate goes stale and the policy sticks forever. Pinning the specific value makes the loop self-correcting: once the retrained model catches that value reliably, its miss rate falls and the policy moves on by itself.

From our run:

| Biased value | Rounds |
|---|---|
| `delay_profile=slow` | 10 |
| `call_order=split` | 4 |
| `batch_size=medium` | 4 |

---

## Governance — self-updating without blind trust

Every candidate model gets a permanent row in `data/registry.db`. Nothing is ever deleted or overwritten. Rollback means flipping a flag, not deleting data.

**The gate has two bars** (both must pass for promotion):

1. **Absolute bar:** Calibrated benign FPR must be ≤ budget + 0.2% slack
2. **Regression bar:** Candidate's suite recall and suite FPR must not be worse than the active version's — measured on one *identical* fresh sample

The second bar matters. Without it, a model that catches all attacks but has 5× the false alarm rate would be promoted automatically.

### Registry from our 20-round run:

```
v1   round 0   [bootstrap]            promoted  ✓  seed detector
v2   round 1   [retrained]            promoted  ✓  held-out recall 0.000 — still promotes (no regression)
v3–v8          [retrained]            promoted  ✓  rounds 2–7 all recall 1.000
v9   round 7   [aggressive_threshold] REJECTED  ✗  FPR 0.1000 breaches 0.0100 budget
v10–v22        [retrained]            promoted  ✓  rounds 8–20
────────────────────────────────────────────────────────────────
ACTIVE: v22   threshold=0.0559
```

`purpleloop rollback --to-version 5` makes v5 active — v22 stays in the trail.

---

## Results from the committed run

Seed `20251013`, 20 rounds, ~100 seconds on a laptop CPU.

| Metric | Value |
|--------|-------|
| Held-out recall, rounds 1–5 | **0.800** |
| Held-out recall, rounds 16–20 | **1.000** |
| Benign FPR, calibrated (all rounds) | **0.0100** — exactly on budget every round |
| Active version FPR on fresh benign | **0.0118** mean |
| Regression-suite recall | **1.000** mean — nothing forgotten |
| Adaptive policy active | 18/20 rounds biased |
| Governance reject | v9 auto-rejected at 0.100 FPR vs 0.01 budget |
| Total events generated | ~70,000 |
| Final corpus size | 3,236 sessions |

### Read this before quoting the curve

The curve rises from 0.800 → 1.000, but it **saturates after one round**. Round 1's variant (`cloudtrail_tampering` + `delay=slow` + `call_order=split`) was missed entirely. After one training example of that style, all 19 subsequent variants were caught. This is a single early miss followed by 19 clean rounds — not a gradual climb across all 20 points.

### Ablation result — negative, and honestly reported

We tested the hypothesis that *technique* features (admin grants, log tampering, sensitive reads) are what drives generalization, not *delivery* features (timing, ordering).

| Arm | Features | Mean held-out recall |
|-----|----------|---------------------|
| `technique_only` | 35 | 0.948 |
| `all` | 67 | 0.906 |
| `delivery_only` | 32 | 0.896 |

**The claim is not supported.** Delivery-only features score 0.896 vs 0.906 for the full set. The synthetic distributions are separable from gross session shape alone — timing and ordering mutations aren't enough to hide the attack. We left this result standing rather than re-partitioning features until it flipped. `docs/RESULTS.md §6` lists the four changes that would make the task hard enough to answer it.

---

## The 60 tests and what they prove

```
tests/
  test_no_leakage.py          8 tests  — the headline number is honest
  test_gen_and_features.py   22 tests  — generators and features are correct
  test_registry_and_gate.py  16 tests  — governance works end to end
  test_corpus_policy_loop.py 14 tests  — the loop machinery runs correctly
```

The most important are in `test_no_leakage.py`:

- **`test_features_ignore_metadata_values`** — confirms features never encode `variant_id`, `label`, `session_id`, or `round`. The model cannot cheat by memorising the variant ID.
- **`test_heldout_eval_refuses_a_variant_already_trained_on`** — confirms `heldout_eval()` raises an `AssertionError` (not a warning — a crash) if the variant is already in the corpus. You cannot accidentally score a seen variant and inflate the number.

---

## Project layout

```
E:\Projects\purple-loop\
│
├── config.yaml                  Every tunable — rounds, seed, FPR budget, model
├── requirements.txt             All dependencies (no AWS, no cloud)
├── Dockerfile                   --network none hard sandbox
├── run.ps1 / Makefile           One-command launcher
│
├── purpleloop/
│   ├── schema.py                Frozen contracts: events, API vocab, knobs, families
│   ├── corpus.py                Append-only parquet store, partitioned by round
│   ├── ablation.py              Delivery vs technique comparison
│   ├── plots.py                 All four charts
│   ├── __main__.py              CLI: run / report / registry / rollback / ablate / reset
│   │
│   ├── gen/
│   │   ├── attacks.py           3 families + 4 mutation knobs
│   │   ├── benign.py            8 benign kinds including 4 hard negatives
│   │   └── common.py            Shared timing + identity helpers
│   │
│   ├── features/
│   │   └── extract.py           Session → 67 features (32 delivery + 35 technique)
│   │
│   ├── model/
│   │   ├── detector.py          Train + score (RandomForest default, LogReg available)
│   │   └── calibrate.py         Threshold → FPR budget
│   │
│   ├── harness/
│   │   └── heldout.py           Score-before-train with leakage assertion
│   │
│   ├── registry/
│   │   ├── registry.py          SQLite audit ledger (immutable rows)
│   │   └── gate.py              Promotion gate (two bars)
│   │
│   └── orchestrator/
│       ├── run.py               The loop — 9 steps per round
│       └── policy.py            Weak-knob adaptive bias
│
├── tests/                       60 tests
│
└── docs/
    ├── RESULTS.md               Full findings, including the negative ablation
    ├── TEAM-PLAN.md             Phase-by-phase ownership and demo running order
    └── figures/                 4 PNG charts (generated by `purpleloop report`)
```

---

## Switching models

The detector supports two classifiers, switchable in `config.yaml` with no code changes:

```yaml
detector:
  model: random_forest        # default — no scaling needed, free feature importances
  # model: logistic_regression  # alternative — interpretable coefficients
  use_isolation_forest: false  # set true to add a benign-only novelty layer
```

RandomForest is the default because it handles mixed-scale features without normalization, produces stable `predict_proba` scores for threshold calibration, and exposes `feature_importances_` which the loop logs after each round.

---

## Security guarantees

| Guarantee | How it is enforced |
|-----------|-------------------|
| No real AWS account needed | No boto3, no AWS SDK anywhere in the codebase |
| No credentials anywhere | `grep -r "aws_access\|aws_secret\|AKIA" .` returns nothing |
| No outbound network calls | `docker run --rm --network none` — even a runaway loop cannot reach anything |
| Held-out measurement is honest | `assert_unseen()` crashes the process if violated |
| Corpus is append-only | `Corpus.append_round()` raises if the round already exists |
| Registry is immutable | SQLite `INSERT` only — no `UPDATE` or `DELETE` on version rows |
| Everything is reproducible | Single seed in `config.yaml`; identical results on every run |

---

## Reproducing from scratch

```powershell
cd E:\Projects\purple-loop

# Reset all generated data (corpus, models, logs, figures)
.\.venv\Scripts\python.exe -m purpleloop reset --yes

# Full run
.\.venv\Scripts\python.exe -m purpleloop run --rounds 20

# Charts + summary table
.\.venv\Scripts\python.exe -m purpleloop report

# Governance audit trail
.\.venv\Scripts\python.exe -m purpleloop registry

# Feature ablation (delivery vs technique, ~5 min)
.\.venv\Scripts\python.exe -m purpleloop ablate

# Roll back to a specific rule version
.\.venv\Scripts\python.exe -m purpleloop rollback --to-version 5
```

A `STOP` file in the project root acts as a kill switch — the loop exits cleanly at the end of the current round.

---

## Team and ownership

| Member | Primary area |
|--------|-------------|
| CS — Cloud Specialist | Repo layout, environment, Dockerfile, benign generator, corpus storage |
| CA — Cybersecurity Analyst | Attack families, knob space, feature relevance review, miss analysis |
| ML1 — ML Engineer #1 | Feature extractor, detector, threshold calibration, held-out harness, ablation |
| ML2 — ML Engineer #2 | Orchestrator, adaptive policy, registry, promotion gate, rollback |

Full phase-by-phase breakdown with exact file ownership: `docs/TEAM-PLAN.md`

Full results write-up including the negative ablation and four suggested improvements: `docs/RESULTS.md`

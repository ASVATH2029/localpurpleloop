# Phase-by-phase plan and ownership

Maps the 4-person plan onto the files in this repo. **CS** = Cloud Specialist,
**CA** = Cybersecurity Analyst, **ML1** = ML Engineer #1 (features, detector,
threshold, held-out harness), **ML2** = ML Engineer #2 (orchestrator, policy,
registry).

Phases 1–3 are parallel workstreams; integration lands at the end of Phase 3.
Timings assume a single working day.

---

## Phase 0 — Kickoff, contracts, environment (~45 min, all hands)

Goal: everyone codes against the same schema, and the repo runs on every laptop.

| Member | Tasks | Lands in |
|---|---|---|
| CS | Repo layout, `requirements.txt`, venv, optional Dockerfile with `--network none` | [requirements.txt](../requirements.txt), [Dockerfile](../Dockerfile), [run.ps1](../run.ps1), [Makefile](../Makefile) |
| CA | Spec the 3 families and the knob ranges; define what benign looks like | [schema.py](../purpleloop/schema.py) |
| ML1 | Draft the feature list and metric definitions (held-out recall, FPR budget) | [features/extract.py](../purpleloop/features/extract.py) |
| ML2 | Draft the round-log and registry schemas; sketch the loop state machine | [registry/registry.py](../purpleloop/registry/registry.py), [run.py](../purpleloop/orchestrator/run.py) |

**Freeze these and do not change them later** — they live in
[schema.py](../purpleloop/schema.py) and `ROUND_LOG_COLUMNS`:

- Event: `timestamp, identity, session_id, eventName, resource, source_ip, params, round, variant_id, label`
- Corpus: 67 features + `session_id, label, round, variant_id`
- Registry: `rule_id, version, created_round, held_out_recall, benign_fpr, suite_recall, promoted, decision, reason, candidate_kind, model_path, rule_body, created_at`
- Knobs: `delay_profile, batch_size, call_order, identity_reuse`

**Exit:** schemas committed, `pytest` collects, `python -m purpleloop --help` works.

---

## Phase 1 — Data layer (~1h45)

Goal: `generate_variant()` and `generate_benign()` work; corpus append is
persistent, partitioned, and refuses to overwrite.

| Member | Tasks | Lands in |
|---|---|---|
| CA *(lead)* | 3 families × 2 seed variants; the 4 mutation knobs; per-round event cap | [gen/attacks.py](../purpleloop/gen/attacks.py) |
| CS | Benign generator **including the 4 hard negatives**; append-only round-partitioned storage | [gen/benign.py](../purpleloop/gen/benign.py), [corpus.py](../purpleloop/corpus.py) |
| ML1 | Sliding-window, timing and n-gram helpers | [gen/common.py](../purpleloop/gen/common.py) |
| ML2 | `variant_id` scheme, knob sampler, registry DDL | [gen/attacks.py](../purpleloop/gen/attacks.py), [registry/](../purpleloop/registry/) |

The hard negatives are CS's most important deliverable: without benign traffic
that looks like each attack along one axis, the FPR budget is free and means
nothing. See the table in the [README](../README.md).

**Exit:** `test_gen_and_features.py` green, including the four hard-negative tests.

---

## Phase 2 — Features and baseline detector (~2h)

| Member | Tasks | Lands in |
|---|---|---|
| ML1 *(lead)* | 67-feature extractor; RandomForest baseline; ROC/PR and importances | [features/extract.py](../purpleloop/features/extract.py), [model/detector.py](../purpleloop/model/detector.py) |
| CA | Review every feature for attacker relevance; flag leakage; add benign-only IsolationForest | [model/detector.py](../purpleloop/model/detector.py) |
| CS | Package extraction as a tested deterministic function | [tests/test_gen_and_features.py](../tests/test_gen_and_features.py) |
| ML2 | Registry module: `add_version`, `promote`, `reject`, `rollback`, `get_active` | [registry/registry.py](../purpleloop/registry/registry.py) |

**Exit:** baseline recall/FPR reported; registry unit tests pass.

---

## Phase 3 — Held-out harness, calibration, promotion gate (~2h)

This is where the project's credibility is won or lost.

| Member | Tasks | Lands in |
|---|---|---|
| ML1 *(lead)* | Score-before-train harness **with the leakage assertion**; per-round threshold recalibration | [harness/heldout.py](../purpleloop/harness/heldout.py), [model/calibrate.py](../purpleloop/model/calibrate.py) |
| ML2 | The promotion gate and the audit-trail message format | [registry/gate.py](../purpleloop/registry/gate.py) |
| CA | A deliberately over-aggressive candidate to force a reject; miss-to-knob labelling | `_propose_aggressive` in [run.py](../purpleloop/orchestrator/run.py) |
| CS | Integration test: one full manual round end to end | [tests/test_corpus_policy_loop.py](../tests/test_corpus_policy_loop.py) |

**Watch for the two bugs we hit here** (both described in
[RESULTS.md](RESULTS.md)):

- The gate must compare candidate and active on **one identical fresh sample**.
  Comparing each version's own calibrated FPR compares nothing, because
  calibration drives both to the budget by construction.
- `heldout_eval()` must **raise**, not warn, when the variant is already in the
  corpus. A warning gets ignored and the headline number silently inflates.

**Exit:** `test_no_leakage.py` green; one forced reject logged.

---

## Phase 4 — Orchestrator and adaptive policy (~2h)

| Member | Tasks | Lands in |
|---|---|---|
| ML2 *(lead)* | The round loop; weak-knob bias policy; round cap, kill switch, structured logging | [run.py](../purpleloop/orchestrator/run.py), [policy.py](../purpleloop/orchestrator/policy.py) |
| ML1 | Clean `train()` / `score()` / `calibrate()` surface; fixed seeds; class imbalance | [model/](../purpleloop/model/) |
| CA | Miss-to-knob attribution; sanity rules keeping mutations realistic | [policy.py](../purpleloop/orchestrator/policy.py) |
| CS | Run it in Docker with no network; one-command launcher | [Dockerfile](../Dockerfile), [run.ps1](../run.ps1) |

**The policy is subtler than it looks** — we got it wrong twice:

1. Rank knob values by a **shrunk** miss estimate, not the raw mean, or one
   noisy round pins the policy for the rest of the run.
2. Bias the specific weak **value**, not just the dimension. Biasing a
   dimension and then drawing some other value within it never tests the
   hypothesis that triggered the bias, so the estimate goes stale and the
   policy sticks anyway.

**Exit:** `purpleloop run --rounds 5` completes unattended; the bias note
appears in `logs/loop.log` and visibly changes what gets generated.

---

## Phase 5 — Full run and validation (~1h30)

| Member | Tasks |
|---|---|
| CS | Launch the 15–20 round run in the sandbox; verify no network/credentials; archive per-round data |
| ML1 | Verify no leakage and the scoring order; **run the ablation** (`purpleloop ablate`) |
| ML2 | Pull the registry evidence: an auto-reject with its reason, plus a rollback |
| CA | Review what was still missed and why; check caught variants are not near-duplicates of training data |

**The four verification checks:**

1. Held-out recall trends upward, or shows clear catches of unseen variants ✔ (0.800 → 1.000)
2. Benign FPR flat and low throughout ✔ (0.0100 calibrated; 0.0118 on fresh benign)
3. At least one logged adaptive-bias decision that changes the next variant ✔ (18/20 rounds)
4. At least one logged auto-reject on false-positive cost ✔ (v9, 0.100 vs 0.01 budget)

**Expect the curve to saturate** and say so. In our run only 1 of 20 rounds was
missed. Report that plainly instead of presenting a 20-point learning curve.

---

## Phase 6 — Results, demo, write-up (~2h)

| Member | Tasks | Lands in |
|---|---|---|
| ML1 | Generalisation curve, FPR overlay, suite chart, ablation chart | [plots.py](../purpleloop/plots.py), `docs/figures/` |
| ML2 | Registry audit table, rollback demo, architecture diagram | [RESULTS.md](RESULTS.md) §4 |
| CA | Threat-model narrative; caught vs missed; limitations | [RESULTS.md](RESULTS.md) §7 |
| CS | README, one-command repro, isolation statement | [README](../README.md) |
| All | Rehearse the 5-minute demo |

### Demo running order

1. **The problem** — hand-written rules catch what their author imagined; a
   slowed-down attack walks past them. Show round 1: recall 0.000.
2. **The loop** — one round, steps 1–9, pointing out that held-out scoring
   happens *before* training.
3. **The curve** — `docs/figures/generalisation_curve.png`, with the honest
   caveat that it saturates after one round.
4. **The governance** — `purpleloop registry`: v9 rejected at 0.100 FPR against
   a 0.01 budget, then `purpleloop rollback --to-version 5`.
5. **What we did not prove** — the ablation's negative result and the four
   changes that would make the task hard enough to answer it. Finishing on a
   real limitation is stronger than overclaiming.

---

## Ownership summary

| Area | Owner | Support |
|---|---|---|
| Sandbox, env, benign data + hard negatives, corpus storage | CS | ML2 |
| Attack families, knob space, feature relevance, miss analysis | CA | ML1 |
| Features, detector, threshold, held-out harness, ablation | ML1 | CA |
| Orchestrator, bias policy, registry, gate, rollback | ML2 | ML1 |

## Risks

| Risk | Mitigation |
|---|---|
| Held-out leakage invalidates the curve | Harness asserts and raises; `test_no_leakage.py` enforces the order (ML1) |
| Curve does not rise | Broaden seeds and features; **report honestly** — see RESULTS.md §6 |
| Gate compares meaningless quantities | Both versions scored on one identical fresh sample (ML2) |
| Policy fixates on one observation | Shrunk estimates + value-level bias (ML2) |
| Docker trouble | Plain Python works; containers are only for the isolation guarantee |
| Scope creep | Hard cap 15–20 rounds; `max_events_per_round` bounds each round |

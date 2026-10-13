# Results

Run: 20 rounds, seed `20251013`, commit as built. ~100 s wall clock on a
laptop CPU, 70,353 synthetic events, 3,236 sessions in the final corpus,
22 rule versions in the registry.

Regenerate with `.\run.ps1 reset && .\run.ps1 run 20 && .\run.ps1 ablate`.

---

## 1. The generalisation curve

Round number vs. recall on that round's brand-new variant, scored **before**
that variant entered the training set.

![generalisation curve](figures/generalisation_curve.png)

| Rounds | Mean held-out recall |
|---|---|
| 1–5 | 0.800 |
| 16–20 | 1.000 |
| all 20 | 0.950 |

Exactly one round was missed, and it is the informative one:

| Round | Family | delay | batch | order | identity | Recall |
|---|---|---|---|---|---|---|
| 1 | cloudtrail_tampering | **slow** | medium | **split** | two_identities | **0.000** |

Round 1's detector had seen only the six hand-authored seed variants, all of
them `burst` or `fast` and `canonical`. Presented with the same technique
delivered slowly and broken into chunks separated by innocuous traffic, it
scored every session below threshold. Once that variant entered the corpus,
every subsequent mutated variant — including `slow`, `jittered`, `reversed`,
`split` and `per_call_identity` deliveries it had never seen — was caught.

**Honest reading: the curve rises, but it saturates after one round.** "0.800 →
1.000" is a single early miss followed by 19 clean rounds, not a gradual climb.
Do not present it as a learning curve with 20 informative points.

## 2. False positives stayed inside budget

| Metric | Value |
|---|---|
| Calibrated benign FPR | 0.0100 in all 20 rounds (budget 0.0100) |
| Active version's FPR on each round's fresh benign | 0.0118 mean |
| Regression-suite recall, candidate | 1.000 mean |

The calibrated figure sits exactly on budget by construction — the threshold is
re-derived each round as the most permissive cutoff that still fits it. The
independent numbers are the second and third rows: measured on benign samples
the model never trained on, the live detector stayed near 1.2% while recall
grew, and it never lost coverage of anything it had already learned.

![regression suite](figures/suite_recall.png)

## 3. The policy adapts, and corrects itself

![knob miss rates](figures/knob_miss_rates.png)

18 of 20 rounds were biased (the first 2 explore uniformly). The policy
rotated across three knob values rather than fixating:

| Biased value | Rounds |
|---|---|
| `delay_profile=slow` | 10 |
| `call_order=split` | 4 |
| `batch_size=medium` | 4 |

A logged decision looks like:

```
policy: biasing toward delay_profile=slow because misses cluster there
        (100% of sessions missed over 1 round)
```

**Two bugs were found and fixed here, and they are worth recording.**

First version: the policy ranked knob values by raw mean miss rate, so a value
seen in a single round outranked one with five rounds of evidence. It pinned
itself to `batch_size` for 15 consecutive rounds on one observation. Fixed with
shrinkage toward the overall miss rate (`PRIOR_WEIGHT = 2.0` in
[policy.py](../purpleloop/orchestrator/policy.py)).

Second version: the policy biased a knob *dimension* but then sampled a
different value within it, so it never gathered evidence about the value that
triggered the bias — the estimate went stale and it stuck anyway. Fixed by
pinning the specific weak **value**, which makes the loop self-correcting: once
the retrained detector catches that value, its estimate falls and the policy
moves on by itself. That is the behaviour in the table above.

## 4. Self-updating is governed, not blind

22 versions, one active, nothing overwritten.

```
v8   round 7  [retrained]            promoted  :: no regression on either axis
                                                  (suite recall 1.000 -> 1.000,
                                                   suite FPR 0.0075 -> 0.0112)
v9   round 7  [aggressive_threshold] REJECTED  :: governance test (10x FPR budget):
                                                  calibrated benign FPR 0.1000
                                                  breaches the 0.0100 budget
                                                  (+0.0020 slack); active kept
ROLLBACK at round 21: v22 -> v5 :: demo: revert to the round-4 rule set
```

The rejected candidate is the deliberate governance test: the same retrained
model with its threshold loosened to a 10× budget. It would have reached the
same suite recall, and the gate refused it on false-positive cost alone. It is
labelled `candidate_kind='aggressive_threshold'` in the registry so nobody
mistakes it for an organic result.

The gate also rejects organically. In an earlier run with the same code,
round 19's retrained candidate was refused because its suite FPR regressed
0.0037 → 0.0163 beyond the 0.01 tolerance. Whether that fires in a given run
depends on benign sampling noise, so the injected test is what guarantees the
mechanism is demonstrated every time.

### One gate bug worth recording

The first implementation compared the candidate's *calibrated* FPR against the
active version's *calibrated* FPR. Both are ≈ the budget by construction, so
that half of the gate compared nothing and would have waved through any
FPR regression. It now compares both versions on one identical fresh sample
(`suite_fpr`), and keeps the calibrated figure only for the absolute budget
check. See [gate.py](../purpleloop/registry/gate.py).

## 5. The ablation: a negative result

The appealing story is that the detector generalises because its features
describe the *technique* (admin-policy grants, log-coverage reduction, reads
concentrated on sensitive data) rather than the *delivery* (timing, burstiness,
ordering, volume). That is testable, so we tested it: same loop, same variant
sequence, policy disabled, only the feature group changes.

![ablation](figures/ablation.png)

| Arm | Features | Mean held-out recall | Mean benign FPR |
|---|---|---|---|
| `technique_only` | 35 | 0.948 | 0.000 |
| `all` | 67 | 0.906 | 0.010 |
| `delivery_only` | 32 | 0.896 | 0.010 |

**The claim is not supported.** Delivery-only features reach 0.896 against
0.906 for the full set — a 0.010 gap. Timing and ordering features alone
generalise about as well as the semantic ones, so the curve in section 1 is
not evidence that technique features are what carry it.

The honest explanation is that the synthetic attack and benign distributions
are separable from gross session shape alone: an attack session differs from
benign traffic in length, API diversity and resource concentration regardless
of how its timing is mutated. The task is too easy for this comparison to
separate the two feature groups.

We left the result as it stands rather than re-partitioning the feature groups
until the verdict flipped. One partition fix *was* applied first, for a reason
independent of the outcome: `frac_<category>` features say *which* calls were
made, so leaving them on the delivery side made the "delivery-only" arm
secretly semantic. After that fix the verdict was unchanged.

## 6. What would make this answerable

The loop machinery is sound; the detection problem is too easy. Four changes,
in the order we would try them:

1. **Make benign traffic overlap the attacks.** Today a `backup_job` has volume
   without sensitivity and an `analyst_sensitive_query` has sensitivity without
   volume. Add benign sessions that are long, bursty, touch sensitive buckets
   *and* involve IAM writes — so no single axis separates the classes.
2. **Make the attacks quieter.** Cap attack sessions to benign-typical length
   and embed them inside otherwise-normal sessions, so `n_events` and
   `resource_concentration` stop being giveaways.
3. **Shrink the seed corpus.** One seed variant per family instead of two, so
   the early model is weaker for longer and the curve has headroom.
4. **Report per-family curves.** `cloudtrail_tampering` is the shortest and
   most distinctive family; pooling it with the other two hides where the
   difficulty actually is.

## 7. Limitations

- **Synthetic events.** Shaped like CloudTrail but generated in-process. No
  claim transfers to real logs without re-validation.
- **Session-level detection.** One `session_id` is one episode. A real attacker
  spreading activity across sessions and days would need a different unit of
  analysis.
- **Sensitivity is declared, not learned.** The feature extractor knows which
  buckets are sensitive from a lexicon in `schema.py`. Realistic (bucket names
  are visible in CloudTrail) but it is domain knowledge handed to the model.
- **The attack generator and the detector share a worldview.** Both were
  written by the same team against the same schema. An attack technique nobody
  thought to generate cannot be measured here.
- **20 rounds is small.** Enough to show the mechanism, not enough to make the
  trend statistically meaningful. Single run, single seed — no confidence
  intervals, and the organic gate rejection in section 4 appearing or not is
  itself evidence of run-to-run variance.

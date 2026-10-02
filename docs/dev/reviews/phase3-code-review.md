# Phase 3 Code Review

QA, 2026-09-25. Scope: the Coder's
- 7aae63c (P3-1 engine);
- 532cb4a (P3-2);
- 0cdc16b (P3-3);
- c429c86 (P3-5);
- f2d22f6 (P3-6);
- d5ae09a (P3-7).

Each is checked against `docs/phase3-plan.md` rev 2. QA's own subtasks are 261bc70 (P3-1 bench), 6d49fb1 (P3-4) and 8ee8daf (P3-8); they need a different role's review (Coder: bench and e2e; Researcher: P3-4).

## Summary

| Subtask | Author | Verdict |
|---|---|---|
| P3-1 split denominator (engine) | Coder | **APPROVED** |
| P3-2 length_z floor, N5 wording | Coder | **APPROVED** (N1) |
| P3-3 stats library | Coder | **APPROVED** (validated by P3-4; N2) |
| P3-5 `rrc eval` / `rrc compare` | Coder | **CHANGES REQUIRED** (R1; N3, N4) |
| P3-6 `rrc train-bc`, `bc:` | Coder | **APPROVED** (N5–N7) |
| P3-7 LeRobot adapter | Coder | **APPROVED** |
| P3-1 bench, P3-4, P3-8 | QA | committed; awaiting Coder / Researcher review |

**Suite:** 219 passed, 3 skipped (gated) in 2 min 23 s. That's inside DoD 11's 2.5 min, but only just (see Risk). `ruff` clean.

## P3-1: R-real-1 (DoD 10)

**Engine (7aae63c):**
- `signals.tv_denominator("split")` = |a_k − a_0| + |a_T − a_k| with k = the farthest frame, which matches brief §6 and the spike bit for bit.
- The `net` formula is kept behind `ScoreConfig.tv_mode` for comparisons.
- The floor rule is unchanged.
- Unit tests cover split ≥ net, split = net on one-way reaches, and out-and-back.

**Bench (QA, 261bc70):**
- `return_home` (clean_variant, 10 per mixed set) is in every mixed set.
- A whole-set low-noise twin (`mixed_low` / `clean_only_low`, σ ~ U(0.002, 0.008)) is built per seed set.
- New bar: return_home flagged ≤ 1/10. On the low profile only that bar binds.
- The power check scores the low set with `tv_mode="net"`.

**Bars against 7aae63c** (full builds, `bench check` → `rrc score` JSON):

| Bar (standard noise) | Target | dev | held-out | QA-private |
|---|---|---|---|---|
| Clean vs. motion junk AUROC | ≥ 0.95 | 0.998 | 1.000 | 1.000 |
| Precision / recall | ≥ 0.90 / ≥ 0.85 | 1.00 / 0.975 | 1.00 / 0.975 | 1.00 / 1.000 |
| Clean flagged (mixed) / clean-only | ≤ 5% / ≤ 5% | 0/60 / 0% | 0/60 / 1% | 0/60 / 2% |
| noise 0.1 / 0.25 / random flagged | ≥ 90% each | 10/10/10 | 10/10/10 | 10/10/10 |
| Hesitation AUROC | ≥ 0.90 | 0.993 | 1.000 | 1.000 |
| Nearmiss flagged | ≤ 20% | 0/10 | 0/10 | 0/10 |
| Report only: wobble / stall / wrong flagged | – | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| **return_home flagged** | ≤ 1/10 | **0/10** | **0/10** | **0/10** |
| **Low-noise return_home flagged (binding)** | ≤ 1/10 | **0/10** | **0/10** | **0/10** |
| Low-noise clean-only (reported) | – | 5% | 3% | 2% |
| Power check: `net` on the low set | ≥ 2/10 on some set | 1/10 | 0/10 | **2/10** |

- All binding bars pass on all three seed sets.
- **The power check is met (private 2/10), but only just.** The `net` false positive depends heavily on the seed set: across 6 fresh bases it flagged 0, 5, 6, 2, 0 and 3 of 10, while split flagged 0/60. Both QA's and the Researcher's return_home generators give the same rates.
- The rate also falls as return_home becomes common: with 40 return_home episodes in the set, `net` flags 0/40, because relative scoring treats them as normal.
- So CI uses a published **regression fixture**, `SEED_SETS["rreal1"]` (16000/16500). It was chosen because the phase 2 bug is visible there (net 7/10, split 0/10), and it's labelled that way in `spec.py`. It isn't a validation set.

## P3-4: statistics validated (QA, 6d49fb1)

All plan bars pass (≈ 9 s). References are scipy 1.17.1 and QA's own code.

| Check | Bar | Result |
|---|---|---|
| (a) Wilson vs. `binomtest(...).proportion_ci("wilson")`, every k, n = 1–60 + up to 200 | ≤ 1e-9 | ≤ 2e-16 |
| (a) Clopper-Pearson vs. scipy `"exact"`, n grid to 200 | ≤ 1e-9 | ≤ 5e-13 |
| (b) `mcnemar_exact` vs. `binomtest(b, b + c, .5)` | ≤ 1e-9 | ≤ 2e-15; b = c = 0 → 1.0 |
| (c) Newcombe vs. QA's own method-10 implementation (2,007 tables, degenerate ones included) | agree | max diff 0 |
| (c) Newcombe exact-enumeration coverage, 53 (p_A, p_B, φ) points | mean 0.94–0.96, worst ≥ 0.90 | n = 50: 0.946 / 0.930; n = 200: 0.949 / 0.946 |
| (d) simulated power (2,000 draws) at the reported MDE | ≥ 0.78 | 0.823 (n 200, p .5, lose 0), 0.808 (lose .1), 0.807 (n 100) |
| (d) MDE is the smallest such Δ | – | MDE − 0.005 < 80% at all 3 points |
| (e) Wilson coverage, 1e-4 grid, p ∈ [0.05, 0.95] | mean 0.945–0.955, worst ≥ 0.91 | n = 20: 0.954 / 0.916; 50: 0.952 / 0.931; 100: 0.951 / 0.927; 200: 0.950 / 0.934 |

## P3-8: headline e2e (QA, 8ee8daf)

Everything goes through the installed `rrc` console script:
1. `bench record` of 60 clean + 10 noise 0.25 + 10 random + 5 hesitation + 5 wrong, seed 30000;
2. `rrc export` → `rrc score`;
3. `train-bc --keep all` vs. `--keep ok-and-success`, train seed 0;
4. `rrc compare`, 200 episodes, eval seed 900000.

Results:
- **A 0/200, B 29/200, McNemar p = 3.7e-9**, with the right verdict, recipe caveat, MDE sentence and success definition.
- The filter kept exactly the quality-ok ∩ sim-success episodes, with no random, noise or wrong-goal episodes.
- Margin over train seeds 0–4: worst p = 6.1e-5.
- The default test takes about 35 s.
- The gated `RRC_E2E_FULL=1` run reproduces brief §4 B vs. E (unfiltered ≤ 5/200, filtered ≥ 100/200, p < 1e-4) in about 65 s.
- **Pending:** "QA runs the README steps in a clean venv" waits for the PM's README section "Evaluate a policy", which isn't written yet.

## Code review notes

**P3-2 (532cb4a):**
- The log-length MAD floor of 0.1 bounds |length_z| for time-capped datasets.
- The N5 marker applies to `ok` rows. Motion signals, flags and bars are unchanged, which the bench confirms.

**P3-3 (0cdc16b):** a faithful port of the verified spike, with edge cases documented and argument checks added. `lgamma` for n > 1000.

**P3-5 (c429c86):**
- The runner is right:
  - env reset from `SeedSequence([eval_seed, i])` with a separate policy child;
  - the env clips actions;
  - an invalid or non-finite action is a clean error;
  - both final-frame and any-step success;
  - a final-state hash per episode.
- Eval targets can't coincide with training targets recorded with the same base seed, because they come from different SeedSequence streams.
- The D2 MDE rule matches the spec: observed `lose` = a_only / A's successes, floored at 0.10, with the 0.5/0.10 fallback; inputs are stored in the JSON.
- Wording:
  - "No detectable difference (…)" with the MDE;
  - the direction is named when significant;
  - the recipe caveat is shown only for two trained checkpoints.
- The report path is protected (refused inside a checkpoint dir; overwrite rules as in `rrc score`).
- Verified by hand:
  - scripted 50/50 and random 0/50;
  - scripted vs. scripted: p = 1, "No detectable difference";
  - scripted vs. random: "B is worse", p = 1.9e-9;
  - bad specs, a non-checkpoint dir, a checkpoint without config, a report inside a checkpoint and `--episodes 0` all fail cleanly;
  - determinism: identical success vectors and final-state hashes for `bc:` at 1 vs. 2 threads.

**P3-6 (f2d22f6):**
- Parity with the spike trainer.
- Seeded torch plus a permutation generator gives the same weight hash at 1 vs. 2 threads (verified by hand).
- `--score-json` must match the dataset's path and frame count.
- The `success` filter uses the label if present, else the sim, and refuses unknown outcomes.
- The checkpoint records the fingerprint, filter, kept episodes and weights SHA-256.
- Loading uses `weights_only=True` and checks the hash (a tampered checkpoint is refused).
- Refusals verified: no `--score-json`, a score JSON from another dataset, a dataset without `observation.environment_state`, and `--overwrite` on a non-checkpoint dir.

**P3-7 (d5ae09a):**
- Inputs are checked against what the sim provides; image and other-robot policies are refused, naming the key.
- CPU only. `reset()` every episode clears ACT's queue (tested).
- lerobot's stdout is redirected, so `--json` stays valid.
- A shared tiny-ACT fixture; no success claim.

## Required changes

1. **R1 (P3-5): say when the verdict and the CI disagree.**
   - The verdict uses the exact McNemar test and the CI uses Newcombe. Near α they can disagree. QA counted **158 of 4,223** paired tables (n = 200; a ∈ {40, 100, 160}; b, c ≤ 40) where they do.
   - Example: `No detectable difference (Δ = +2.5 pts, 95% CI +0.2 to +5.0; McNemar p = 0.063)`. Without an explanation, that reads as self-contradictory.
   - When `significant != (CI excludes 0)`, `compare` must add one sentence, e.g. *"The CI and the exact test disagree at the margin (the exact test is more conservative); treat this result as borderline."* Record `"borderline": true` in the JSON `paired` block, and add a test with one of the tables above.

## Non-blocking

- **N1 (P3-2):** HARD rows can list sub-threshold reasons (3 < z ≤ 3.5) without the "(below flag threshold)" marker. Apply it whenever `combined ≤ flag_z`.
- **N2 (P3-3):** `wilson(n, n)` returns an upper bound of 1 − 1e-16, while `wilson(0, n)` returns exactly 0.0. Clamp for symmetry, since the docstring says the interval touches 1.
- **N3 (P3-5):** the MDE sentence is also printed for significant results, and for policies that aren't checkpoints ("disagreement between the checkpoints" for scripted vs. random). Print it only when not significant, and say "policies".
- **N4 (P3-5):** only `PolicySpecError`, `ImportError`, `ValueError` and `OSError` are caught around the rollouts. A torch `RuntimeError` in `act()` (e.g. an architecture that doesn't match the sim) would print a traceback. Validate `arch.inputs == 9` and `arch.outputs == 6` at load, or catch `RuntimeError`.
- **N5 (P3-6):** `--overwrite` deletes the old `model.pt` / `rrc_policy.json` *before* training, so a failed training run loses the previous checkpoint. Write to a temp dir and swap at the end.
- **N6 (P3-6/P3-5):** a corrupt `model.pt` gives a clean `rrc: error:`, but it includes torch's multi-line `weights_only` advice. Trim it to the first line.
- **N7 (P3-6):** `torch.set_num_threads` is process-global. That's fine for the CLI; note it in the docstring for library callers.

## Risk

The suite takes 2 min 23 s against a 2.5 min budget. The biggest items are:
- the P3-8 e2e (35 s);
- the small bench build (15 s);
- the tiny-ACT fixture (8 s);
- train-bc determinism (9 s).

If P3-5 R1 or the README tests push it over, gate the P3-8 default test behind `RRC_E2E=1` and run it in QA's acceptance instead. Don't shrink its episode count: the p < .01 margin depends on n = 200.

## Re-review

QA, 2026-09-25. Scope:
- `94aeac4` (P3-5 R1 + N1–N7);
- the PM's README section "Evaluate a policy" (uncommitted);
- the Coder's and Researcher's reviews of QA's own work (`docs/reviews/phase3-qa-work-review.md`: P3-1 bench, P3-8 and P3-4 approved).

**Suite:** 234 passed, 3 skipped in 2 min 16 s here (the coordinator measured 144 s). DoD 11 is < 2.5 min.

**Fixes verified:**
- **R1:** `Comparison.borderline` = significant ≠ CI-excludes-0. It flags exactly QA's **158/4,223** tables. The sentence and `paired.borderline` are shown, and there's a test.
- **N1:** the "(below flag threshold)" marker now also applies to HARD rows.
- **N2:** `wilson(30, 30)` = (0.886, 1.0) and `wilson(0, 30)[0]` = 0.0.
- **N3:** "policies" unless both sides are checkpoints. The MDE sentence stays on significant results. **Accepted:** DoD 2 requires it on every compare, and the P3-8 e2e asserts it.
- **N4:** a checkpoint with 12 inputs gives "checkpoint maps 12 inputs to 6 outputs; so101_reach needs 9 …". `RuntimeError` in rollouts is a clean error.
- **N5:** a failed `--overwrite` leaves the old checkpoint intact.
- **N6:** a corrupt `model.pt` gives a one-line `rrc: error`.
- **N7:** docstring updated.

**README, run verbatim in a fresh uv venv** (the working tree copied, venv deleted afterwards):
- `pip install -e ".[dev,eval]"`: 23 s from cache, **5.4 GB** (CUDA torch on Linux). `rrc eval scripted` works without pyarrow or lerobot.
- `".[dev,eval,score,lerobot]"`: **10 GB** total.
- The demo (`bench record` → export → score → 2× train-bc → eval → compare) takes **51 s** (the README says about 45 s). Results:
  - **A 0/200, B 29/200, Δ +14.5 pts (CI +9.9 to +20.0), McNemar p = 3.7e-9**, exactly as documented;
  - verdict, MDE (9.0 pts), recipe caveat and success definition are all printed;
  - JSON reports are written to the working dir.
- Content check:
  - the success definition, the Wilson/McNemar/Δ/MDE reading guide, "No detectable difference ≠ equal", checkpoint vs. recipe (10–15 pts, ≥ 3 seeds), and the D5 per-junk-type wording all match the plan and the measurements;
  - determinism is stated as same-machine only;
  - there's no ACT success claim.

**Rulings:**
1. **Don't gate the P3-8 e2e.** Plan P3-8 acceptance puts DoD 9 "within ~60 s in the default suite", and it's the phase's headline regression guard. The suite is under budget (135–144 s). If a later addition breaks the budget, gate the slower non-headline tests first (the train-bc determinism check at 9 s, the tiny-ACT setup at 8 s) rather than shrink or gate the e2e; its margin depends on n = 200. **No test change, no commit.**
2. **A README demo that depends on `python -m robot_report_card.bench record` is acceptable for phase 3.** It ships in the package and is tested (`bench record` + parse tests). It's the only honest way to build the pinned mix today: `rrc record` has no hesitation policy, and its `wrong` still uses the mirrored target, whose joint-limit artifact phase 2 removed from the bench. Two follow-ups for the backlog:
   - (a) give `rrc record` the bench generators, or add a `--mix` option, and fix its `wrong` policy to use an independent goal, then point the README at it;
   - (b) until then, `bench record` is a supported command and its flags shouldn't change without a README update.

**Non-blocking README notes** (PM, next docs pass):
- give the Linux install size (about 5–10 GB with CUDA torch) and the CPU wheel option (`--index-url https://download.pytorch.org/whl/cpu`), as the plan's risk table promised;
- "about 45 s" → "about 50 s".

### Final verdicts

| Subtask | Verdict |
|---|---|
| P3-1 split denominator (engine) | **APPROVED** |
| P3-1 bench (QA; reviewed by the Coder) | **APPROVED** |
| P3-2 scoring cleanup | **APPROVED** |
| P3-3 stats library | **APPROVED** |
| P3-4 stats validation (QA; reviewed by the Researcher) | **APPROVED** |
| P3-5 `rrc eval` / `rrc compare` | **APPROVED** (R1 fixed) |
| P3-6 `rrc train-bc` / `bc:` | **APPROVED** |
| P3-7 LeRobot adapter | **APPROVED** |
| P3-8 e2e (QA; reviewed by the Coder) | **APPROVED** |
| P3-8 README (PM) | **APPROVED** (non-blocking notes above) |

**PHASE 3 APPROVED.** Close-out, owned by the PM:
- commit the README and `docs/reviews/phase3-qa-work-review.md`;
- update STATUS with the bar tables, P3-4 results and backlog items (a) and (b);
- the Mac milestone check (D6).

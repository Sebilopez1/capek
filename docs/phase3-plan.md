# Phase 3 Plan — Policy Evaluation

Author: PM · Date: 2026-09-24 · Status: rev 2, QA changes applied (see end)
Inputs: `docs/phase3-research-brief.md`, spikes `docs/spikes/phase3_{stats,bc_eval,lerobot_adapter,tv_fix}.py`,
`STATUS.md` (DoD 8 real-data result, backlog). Sebi's scope: a tiny policy trained on our sim data (a real A vs B),
plus LeRobot checkpoints through a plug-in adapter smoke-tested on one small example. All evaluation runs in `so101_reach`.

## Goal

Answer two questions honestly: **"How good is this checkpoint?"** (success rate with a CI) and **"Is B better than
A?"** (paired test, CI on the difference, and what the test *could* have detected). Show the phase 2 link with
numbers: filtering data with `rrc score` + outcome gives a measurably better policy.
Out of scope: early stopping, real-robot eval, image policies, Hub download, multi-task eval, multi-seed recipe command, HTML.

## Definition of Done

1. **`rrc eval POLICY`** runs N seeded rollouts (default `--episodes 200`, `--eval-seed 900000`, episode i reset from
   `SeedSequence([eval_seed, i])`). It prints `policy | n | successes | rate | Wilson 95% CI | median final error`, and
   the success definition verbatim: *"success = gripper tip within 2 cm of the target on the final frame (90 frames,
   3 s); LeRobot's eval counts success at any step"*. `success_any_step` is also reported. POLICY is `scripted`,
   `random`, `bc:<dir>` or `lerobot:<dir>`.
2. **`rrc compare A B`** runs both policies on the *same* seeds in one process and prints both rates with Wilson CIs,
   discordant counts, exact McNemar p (α .05), Δ = p_B − p_A with a Newcombe (method 10) paired 95% CI, and the MDE
   (D2): `mde_paired(n, p̂_A, lose)` with `lose` = b / (a + b) (share of A's successes that B failed) floored at 0.10;
   if A has no successes, p_A = 0.5 and lose = 0.10. It's always printed as *"With n = {n} and this much disagreement
   between the checkpoints, this test detects differences of about X points or more with 80% power."*, where {n} is
   the actual n. The inputs (n, p̂_A, lose, floored?) go in the JSON. A non-significant result starts *"No detectable
   difference (Δ … CI … p …)."* A significant one names the direction ("B is better/worse than A"). The output never says
   "no difference" or "equivalent".
3. **Recipe caveat:** when both sides are trained checkpoints, `compare` prints *"This compares two fixed checkpoints;
   the CI covers rollout noise only. Retraining the same recipe with another seed moved success by 10–15 points on this
   task. A claim about data or training settings needs ≥ 3 training seeds per side with a consistent direction."*
4. **`rrc train-bc DATASET --out DIR`** trains the MLP (9→256→256→6, predicts `action − state`) on an rrc export on CPU.
   `--keep all|quality-ok|success|ok-and-success` filters via an `rrc score` JSON (`--score-json`) and outcome evidence.
   The checkpoint dir holds `model.pt` + `rrc_policy.json` (arch, normalization, obs keys, train seed, epochs,
   dataset fingerprint, filter, kept episode indices, weights sha256).
5. **LeRobot adapter** (`lerobot:<pretrained_model dir>`): `reset()` / `act(obs)` via `get_policy_class` + saved
   pre/post-processors, CPU only. Smoke-tested on a tiny state-only ACT trained inside the test (not checked in).
   Policies needing inputs the sim lacks (images) are refused clearly. **No success bar is claimed for ACT.**
6. **Output:** table + JSON (`schema_version: 1`, `kind: eval|compare`), with the same `--json-out` / `--overwrite` /
   `--json` behavior as `rrc score`. The JSON records per-episode seed, success, success_any_step, final error and
   final-state sha256, the eval seed list, policy metadata, and machine info (platform, torch/mujoco versions, threads).
7. **Determinism (same machine):** `rrc eval` twice in separate processes (1 vs 2 torch threads) → identical success
   vectors and final-state hashes; `rrc train-bc` twice with one seed → identical weight hashes. The README says
   cross-machine reruns are new samples, not replays.
8. **Statistics validated by QA** against independent references (P3-4).
9. **Headline e2e passes (pinned, D4):** record 60 clean + 10 noise 0.25 + 10 random + 5 hesitation + 5 wrong (bench
   generators, fixed seeds) → export → score → `train-bc --keep all` (A) vs `--keep ok-and-success` (B), train seed 0 →
   `compare --episodes 200 --eval-seed 900000` → B > A, McNemar p < .01, correct wording. A platform that misses is reported, not retuned.
10. **R-real-1 fixed (D3):** on dev, held-out and QA-private, the standard sets still pass every DoD 4 bar, and
    return_home is flagged ≤ 1/10 in **both** the standard and the low-noise mixed set. A power check shows the phase 2
    `net` formula flags ≥ 2/10 return_home on at least one low-noise set, so the bar can fail.
11. `pytest` default suite < 2.5 min. Slow tests are gated by env vars (`RRC_BENCH_FULL=1`, `RRC_E2E_FULL=1`, `RRC_ACT_FULL=1`).

## Design decisions

- **Policy protocol:** `reset(seed_seq)` / `act(obs) -> action`, with an obs dict holding `observation.state` and
  `observation.environment_state`. The existing scripted/random policies implement it, so eval tests run without torch.
- **Runner:** `robot_report_card/eval/runner.py` reuses `make_env("so101_reach")`. Actions are clipped to the action range,
  and the policy RNG gets its own child seed. Success is the env's final-frame `success`.
- **Stats:** `robot_report_card/eval/stats.py`, pure Python + numpy, ported from `phase3_stats.py`:
  `wilson`, `clopper_pearson` (`--ci clopper-pearson`), `mcnemar_exact`, `newcombe_paired_diff`, `mde_paired`. No scipy at runtime.
- **Extras:** new `eval` = `mujoco>=3.2`, `torch>=2.2` (for `bc:`). `lerobot:` needs the existing `lerobot` extra.
  `rrc eval scripted` needs only `sim`.
- **Filters in `train-bc`** reuse the phase 2 reader and score JSON. `quality-ok` = quality `ok`; `success` =
  `outcome_label` if present, else `outcome_sim`. Episodes with unknown outcome are refused for `success` filters, with a clear error.

## Subtasks (ordered)

### P3-1 R-real-1: split denominator + return_home group
- **Scope:** Coder: `action_tv_ratio` denominator = |a_k − a_0| + |a_T − a_k|, k = farthest frame from the start
  (brief §6); floor unchanged. QA (bench, D3): a `return_home` group (10; out U(0.8, 1.2) s, hold U(0.1, 0.4) s, back to
  start) in **every** mixed set, plus a **low-noise mixed set** per seed set (every σ-drawing group at σ ~ U(0.002, 0.008);
  noise groups keep their fixed σ) with its own return_home ×10. The class `clean_variant` is excluded from P/R/AUROC.
- **Acceptance:** DoD 10 (bars on the standard sets, return_home ≤ 1/10 on standard and low-noise, the `net`-formula
  power check in the bench tests); low-noise clean-only reported, not barred; unit test: split ≥ net on random traces,
  split = net on one-way reaches; before/after table in STATUS.
- **Depends on:** nothing. **Does:** Coder (engine), QA (bench + bar). **Reviews:** Researcher (engine), Coder (bench).

### P3-2 Scoring cleanup: R-real-2 and N5
- **Scope:** `length_z` uses a floored MAD (or the length percentile) so a common time cap can't give z ≈ −600;
  `ok` rows mark reasons with 3 < z ≤ 3.5 as "(below flag threshold)".
- **Acceptance:** a test with 90% of episodes at one length gives |length_z| < 20; there's a N5 wording test; phase 2 bars are unchanged.
- **Depends on:** P3-1 (same files). **Does:** Coder. **Reviews:** QA.

### P3-3 Stats library
- **Scope:** `eval/stats.py`, a straight port of `phase3_stats.py` (D1: all five functions verified by QA), with docstrings citing the method (Wilson 1927; Newcombe 1998 method 10;
  exact McNemar = two-sided binomial on discordant pairs).
- **Acceptance:** edge cases k = 0, k = n, b + c = 0, and n = 1 give finite, documented results; no scipy import; unit tests.
- **Depends on:** nothing (parallel with P3-1/2). **Does:** Coder. **Reviews:** QA (via P3-4).

### P3-4 Statistical correctness (QA-owned validation)
- **Scope:** `tests/test_p3_4_stats_validation.py`, QA's own references (scipy in the dev venv only):
  (a) Wilson and Clopper-Pearson match `scipy.stats.binomtest(k, n).proportion_ci(method="wilson"/"exact")` to 1e-9,
  k, n ≤ 200; (b) `mcnemar_exact(b, c)` matches `binomtest(b, b + c, 0.5).pvalue`; (c) Newcombe: QA's independent
  implementation from the paper agrees, and exact enumeration of the paired multinomial on a (p_A, p_B, φ) grid at
  n = 50/200 gives mean coverage 0.94–0.96, worst ≥ 0.90; (d) `mde_paired`: simulated power at the reported MDE ≥ 0.78
  (2,000 draws, 3 grid points); (e) Wilson coverage on a 1e-4 grid over p ∈ [0.05, 0.95], n = 20–200: mean 0.945–0.955,
  worst ≥ 0.91.
- **Acceptance:** all pass in < 20 s (bigger grids are gated); a failure goes back to P3-3.
- **Depends on:** P3-3. **Does:** QA. **Reviews:** Researcher.

### P3-5 `rrc eval` + `rrc compare`
- **Scope:** policy protocol, runner, the two commands, the table + JSON (DoD 1, 2, 3, 6), and policy spec parsing.
- **Acceptance:** `eval scripted --episodes 50` = 50/50, `random` ≤ 5/50; `compare scripted random` significant with
  correct discordant counts; `compare scripted scripted` → "No detectable difference" + MDE, p = 1; cross-process
  determinism test (DoD 7, eval part); success definition in table and JSON; existing `--json-out` refused as in `rrc score`.
- **Depends on:** P3-3. **Does:** Coder. **Reviews:** QA.

### P3-6 `rrc train-bc` + `bc:` policy
- **Scope:** port the MLP trainer from `phase3_bc_eval.py`: seeded torch + DataLoader, CPU, `torch.set_num_threads`
  from `--threads`, filters, checkpoint format (DoD 4), `bc:` loader.
- **Acceptance:** trains on a 60-episode export in < 10 s; same-seed retrain → identical weight hash across processes
  and 1 vs 2 threads; a clean-data checkpoint beats `random` in `compare` (recipe caveat shown); a bad filter (unknown
  outcome, empty result) gives a clean error.
- **Depends on:** P3-5. **Does:** Coder. **Reviews:** QA, Researcher (training setup matches the brief).

### P3-7 LeRobot adapter
- **Scope:** `eval/lerobot_adapter.py` from the spike (CPU, `reset()` per episode, input-feature check, pre/post-processors).
  A session fixture trains a tiny state-only ACT via `lerobot_train` (`--steps` ≈ 20, `push_to_hub=false`, wandb off)
  on a small rrc export; gated `RRC_ACT_FULL=1` repeats the brief's 300-step run.
- **Acceptance:** `rrc eval lerobot:<dir> --episodes 5` writes a valid report; `reset()` clears the action queue
  (tested); an image-input config is refused naming the key; skipped cleanly without lerobot; no success assertion.
- **Depends on:** P3-5. **Does:** Coder. **Reviews:** Researcher (lerobot API), QA.

### P3-8 Headline e2e, README, STATUS
- **Scope:** QA: `tests/test_p3_8_e2e.py` via the installed `rrc`, exactly the DoD 9 recipe (bench generators,
  `--append`); the gated full version reproduces brief §4 B vs E. PM: README "Evaluate a policy" (definition, n and MDE,
  checkpoint vs recipe, determinism scope) + STATUS, with the D5 per-junk-type wording: random/high-noise actions and
  wrong-goal demos hurt the policy, hesitation didn't; the scorer removes the first two and outcome evidence removes wrong
  goals; E > A is a 3/3-seed recipe result, not a significant single-checkpoint one.
- **Acceptance:** DoD 9 (p < .01, correct direction and wording) within ~60 s in the default suite. QA runs the README steps
  in a clean `[dev,sim,score,eval]` venv.
- **Depends on:** P3-1..P3-7. **Does:** QA (e2e), PM (docs). **Reviews:** Coder reviews QA's e2e; QA reviews PM's docs.

## Risks

| Risk | Mitigation |
|---|---|
| The small e2e may not reach p < .01 | Mix pinned (D4): QA measured p ≤ 6e-5 on 5 training seeds; without random junk it failed on one seed |
| Cross-machine rollouts differ (Mac arm64 vs Linux x86; Accelerate vs OpenBLAS) | Only same-machine determinism is tested and promised; the JSON records machine info; Sebi runs the spike on the Mac |
| Users read a checkpoint comparison as a recipe claim | Recipe caveat printed and in the JSON; README section |
| Our final-frame success ≠ LeRobot's any-step number | Definition printed; `success_any_step` reported alongside |
| torch on Linux pulls CUDA wheels (GBs) | `eval` is an optional extra; README points to the CPU wheel index; the Mac is unaffected |
| LeRobot API churn; Hub checkpoints untested; ACT is weak on CPU | Pinned 0.4.4; adapter smoke test only; image policies refused; no ACT bar |
| The R-real-1 fix shifts bars (low-noise clean-only was at 5%) | Full bars on 3 seed sets in P3-1; low-noise clean-only reported, not barred, until real noise levels are known; report misses, don't retune on held-out |
| Suite time: 47.6 s today + e2e ≈ 60 s + ACT fixture ≈ 15–25 s + stats ≈ 20 s | Session-scoped ACT fixture; gate it if the total passes 2.5 min |
| Test time creeps past 2.5 min | Session-scoped fixtures, small defaults, env-gated full runs; QA reports suite time at close-out |
| Scope (8 subtasks) | Every piece has a working spike; if time runs short, P3-7 (adapter) ships last and P3-2 can slip to backlog |

## Defaults (Sebi can override at milestone)

1. n = 200 rollouts per checkpoint; fixed n, no early stopping.
2. Success is final-frame (the arm must hold at the target). Any-step success is reported next to it.
3. torch comes via a new optional `eval` extra; `rrc eval scripted` works without it.
4. The ACT test checkpoint is trained inside the test, not checked into the repo (avoids a ~6 MB binary).

## Needs Sebi's call

- Nothing blocks the build. **Milestone check (D6):** at the end of phase 3, Sebi runs
  `python docs/spikes/phase3_bc_eval.py ~/rrc-p3-spike` on the Mac (~3 min, the existing `[lerobot]` env) and re-scores the
  DoD 8 dataset (ep 7 should now be `ok`, ep 9 still flagged). The numbers go to STATUS and need not match Linux bitwise.

## QA Review (phase 3)

QA, 2026-09-25. **Verdict: CHANGES REQUIRED (spec-level only).** The Coder can start P3-3, P3-5 and P3-6 now. The changes affect P3-1, P3-4, P3-8, DoD 2, DoD 3, DoD 9 and DoD 10, plus the Mac item. Measurements are in the brief's QA review.

**Findings:**
1. **DoD 10 / P3-1: the planned return_home bar can't fail.** With clean at standard noise and only return_home low-noise, as P3-1 specifies, the *unfixed* formula also flags 0/10 on dev, held-out and private. The false positive only appears when the **whole dataset** is low-noise (net: 5/3/2 of 10). The bar has to be judged where the bug is visible.
2. **P3-4(e) "Wilson worst ≥ 0.92" fails on a fine grid:** it's 0.916 at n = 20.
3. **DoD 2: "the MDE at this n" is undefined.** At the brief's assumed 10% disagreement it is 9 pts. At the disagreement actually observed between two checkpoints (22%) it is 12.5 pts.
4. **DoD 9: p < .01 is deterministic on one machine but has no margin without random junk.**
   - QA measured the small e2e (60 clean + 30 junk, eval 200) over 5 training seeds.
   - **With 10 random + 10 noise 0.25 + 5 hesitation + 5 wrong:** p ≤ 6e-5 on every seed (unfiltered 0–3/200, filtered 15–39/200).
   - **Without random** (15 noise / 8 hesitation / 7 wrong), or with **wrong only**: seed 4 gives p = .24 and .42.
   - The junk mix has to be pinned.
5. **DoD 3: "~11 points" is one variant.** Measured seed-to-seed ranges are 9.5–15 points.
6. **Plan vs. brief:** consistent, apart from the items above and the ACT fixture (the plan trains it in the test; the brief proposes checking it in). The plan's choice stands.
7. **Scope:** 8 subtasks fit one session because each has a working spike. The risk is suite time (DoD 11 < 2.5 min):
   - today's suite: 47.6 s;
   - small e2e: ≈ 60 s;
   - the `lerobot_train` ACT fixture: ≈ 15–25 s;
   - stats validation: ≈ 20 s.

   Keep the ACT fixture session-scoped and gate it if the total goes over.
8. **Success wording** (DoD 1, final frame vs. LeRobot any-step, verified in the 0.4.4 source) **and recipe wording** (DoD 3) are honest. Keep them verbatim.
9. **Mac run: not a blocker.**
   - Nothing in P3-6 depends on Mac numbers. Determinism is promised same-machine only, and timings only affect UX.
   - torch already works on Sebi's Mac (the phase 1 `[lerobot]` env, 1.8 GB).
   - It becomes a milestone check.

**Required changes:**
- R1 (DoD 10, P3-1): replace the bench spec with decision D3.
- R2 (P3-4(e)): Wilson coverage bar = mean 0.945–0.955 and worst ≥ 0.91 on a 1e-4 grid over [0.05, 0.95], for n = 20–200.
- R3 (DoD 2): define the printed MDE per decision D2.
- R4 (DoD 9, P3-8): pin the e2e recipe per decision D4.
- R5 (DoD 3): the caveat reads "moved success by 10–15 points on this task".
- R6 (Needs Sebi's call): the Mac spike run becomes a **milestone check** at the end of phase 3, not a blocker before P3-6.

**Reconciled decisions (binding for the Coder and QA):**
- **D1 Stats (P3-3):** port `phase3_stats.py` as is: `wilson`, `clopper_pearson`, `mcnemar_exact`, `newcombe_paired_diff` and `mde_paired` are all verified. No scipy at runtime. Edge cases per P3-3.
- **D2 MDE in `compare`:** `mde_paired(n, p̂_A, lose)`:
  - `lose` = observed share of A's successes that B failed (b / (a + b)), floored at 0.10;
  - if A has no successes, use `lose` = 0.10 with p_A = 0.5.

  Print it as *"With n = 200 and this much disagreement between the checkpoints, this test detects differences of about X points or more with 80% power."* Store the inputs in the JSON.
- **D3 R-real-1 (P3-1):**
  - Coder: split denominator (brief §6), floor unchanged.
  - QA, bench:
    - add a `return_home` group (10) to every mixed set;
    - add a **low-noise mixed set** per seed set (every σ-drawing group at σ ~ U(0.002, 0.008); noise groups keep their fixed σ), with its own return_home ×10;
    - the class `clean_variant` is excluded from the P/R/AUROC bars.
  - Binding bars:
    - all DoD 4 bars on the standard sets (dev, held-out, QA-private), unchanged;
    - return_home flagged ≤ 1/10 on **both** the standard and the low-noise sets;
    - a power check in the bench tests: the phase 2 `net` formula flags ≥ 2/10 return_home on at least one low-noise set.
  - Low-noise clean-only is reported, not barred (dev 5%, held-out 3%, private 2%). Real noise levels decide later.
  - Before/after table in STATUS.
- **D4 Headline e2e (P3-8):**
  - record 60 clean + 10 noise 0.25 + 10 random + 5 hesitation + 5 wrong (bench generators, fixed seeds);
  - `--keep all` vs `--keep ok-and-success`, train seed 0, eval 200 at `--eval-seed 900000`;
  - assert p < .01, B > A and the wording.
  - The gated full version reproduces brief §4 B vs E.
  - If a platform misses p < .01, report it; don't retune.
- **D5 Wording (README / STATUS / milestone):** use the per-junk-type result from the brief's QA review:
  - random and high-noise actions and wrong-goal demos hurt the policy; hesitation didn't;
  - the scorer removes the first two, and outcome evidence removes wrong goals;
  - E > A is a 3/3-seed recipe result, not a significant single-checkpoint one.
- **D6 Mac:** milestone check (R6). Sebi's numbers go to STATUS; nothing waits for them.

### Changes applied (PM, 2026-09-25, rev 2)
- R1/D3: DoD 10 and P3-1 rewritten. return_home is in every mixed set, there's a low-noise mixed set per seed set, `clean_variant` is excluded from P/R/AUROC, return_home ≤ 1/10 on standard and low-noise, the `net` power check, and low-noise clean-only is reported only.
- R2: P3-4(e) Wilson bar = mean 0.945–0.955, worst ≥ 0.91 on a 1e-4 grid, n = 20–200.
- R3/D2: DoD 2 defines the MDE from the observed disagreement (floor 0.10), prints the actual n, stores the inputs in the JSON, and names the direction when significant.
- R4/D4: DoD 9 and P3-8 pin the e2e mix, train seed and eval seed.
- R5: the recipe caveat says "10–15 points". D5 per-junk-type wording goes in P3-8 docs.
- R6/D6: the Mac run is a milestone check; nothing waits for it. D1 is noted in P3-3. Suite-time risk row added.

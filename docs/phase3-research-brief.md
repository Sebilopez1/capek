# Research Brief: Phase 3 Policy Evaluation

Researcher, 2026-09-24. Everything was measured in this container (Linux x86, **2 CPU threads**, Python 3.11, torch 2.10 CPU, mujoco 3.14, lerobot 0.4.4) unless marked otherwise. **[V]** = checked today (source, web search/fetch or a run); **[M]** = from memory, unverified.

Spikes in `docs/spikes/`:
- `phase3_stats.py`: CIs, paired tests, power/MDE, coverage study.
- `phase3_bc_eval.py`: record → export → score/filter → train BC → paired eval.
- `phase3_lerobot_adapter.py`: runs a LeRobot checkpoint in `so101_reach`.
- `phase3_tv_fix.py`: R-real-1 fix on benchmark v2 plus a return-home group.

## Recommendations

1. **Success-rate CI: Wilson score (95%).** It's closed form (no scipy), and its coverage stays close to nominal at small n (mean ≈ 0.95; worst ≥ 0.91 for p in [0.05, 0.95] and n ≥ 20 on a fine grid; lower near 0 or 1, e.g. 0.84 at n = 20). Show Clopper-Pearson only on request (conservative).
2. **Regression test: a paired design.** Both checkpoints run on the *same* eval seeds, with the **exact McNemar** test on discordant pairs. Report the effect as **p_B − p_A with a Newcombe (1998, method 10) paired 95% CI**. That's all pure Python and already in `phase3_stats.py`. Use a **fixed n = 200 episodes per checkpoint** by default; skip sequential stopping in phase 3.
3. **"No detectable difference" must carry its MDE.** At n = 200 paired, the smallest improvement we can reliably detect (80% power, α = .05) is **about 9–13 points, depending on how often the two checkpoints disagree** (12.5 at the disagreement we observed for A vs E, where B lost 22% of A's successes), or 4 points only if B never loses an episode A won. `compare` should print the MDE *at the observed disagreement* next to every non-significant result.
4. **Say what's being compared.** A *checkpoint* is fixed weights, and our CIs cover rollout noise only. A *recipe* (data or training settings) also has **training-seed variance, which we measured at 10–15 points** across seeds 0/1/2 (A: 44.5–56%, C: 19–34%). Recipe claims need at least 3 training seeds with the direction consistent across them.
5. **Repeatability works in our stack, bitwise on one machine.** Same seeds give the same success vector and the same final-state hash across processes, 1 vs 2 torch threads, and a retrain with the same seed. Don't promise bitwise equality across machines (Mac vs Linux). Store per-episode results so reruns can be diffed.
6. **Tiny policy: a torch MLP BC** on `[observation.state, environment_state] → action − state`. It trains in **7–11 s** on CPU, and 200 paired eval rollouts take **about 7 s**. It produces real A/B differences and **a measured data-quality → policy-quality link** (§4, per-junk-type results from QA):
   - random actions are catastrophic (+30 episodes gives 0%), high-noise demos (14%) and wrong-goal demos (28%) hurt, and hesitation doesn't (57% vs 52% clean);
   - the scorer removes the first two, and only outcome evidence removes wrong goals;
   - scorer ok AND success (60%) beats clean-only (52%) on 3/3 seeds, a recipe-level result that is not significant at the checkpoint level (p = .24).
7. **LeRobot plug-in: a 2-method adapter** (`reset()`, `act(obs)`). It uses `get_policy_class(cfg.type).from_pretrained(path)` plus the saved pre/post-processors, and was verified on a CPU-trained, state-only **ACT**. ACT *trains* on CPU (300 steps in 54 s), but **it doesn't reach the target within minutes** (0/100 at 2,000 steps, 6 min). Use ACT as an adapter smoke test and the MLP for A/B.
8. **R-real-1 fix: "split" denominator** for `action_tv_ratio`, i.e. |a_k − a_0| + |a_T − a_k| where k is the farthest point from the start.
   - By the triangle inequality it is never smaller than today's |a_T − a_0|, so it can only lower the *raw* ratio (z-scores still move with the dataset median/MAD).
   - Return-home false flags go from **5/10 (dev) and 3/10 (held-out) to 0/10** (low-noise setting), with **all DoD 4 bars still passing** on dev (and held-out).

## 1. Confidence interval for a success rate

Exact coverage of nominal 95% intervals, computed over p ∈ [0.05, 0.95] (`python phase3_stats.py`), shown as mean / worst case, width:

| n | Wilson | Clopper-Pearson | Jeffreys |
|---|---|---|---|
| 20 | 0.953 / 0.925* , w 0.33 | 0.975 / 0.959, w 0.37 | 0.948 / **0.896**, w 0.32 |
| 50 | 0.952 / 0.932, w 0.21 | 0.967 / 0.953, w 0.23 | 0.946 / **0.885**, w 0.21 |
| 100 | 0.950 / 0.933, w 0.15 | 0.964 / 0.954, w 0.16 | 0.948 / 0.935, w 0.15 |
| 200 | 0.950 / 0.938, w 0.11 | 0.960 / 0.952, w 0.11 | 0.949 / 0.941, w 0.11 |

- \* Grid step 0.01. On a 1e-4 grid (QA), Wilson's worst case is **0.916** (n = 20), 0.931 / 0.927 / 0.934 (n = 50 / 100 / 200). Near the edges (p = 0.005–0.05, which is common for us) it falls to 0.838 at n = 20 and 0.916 at n = 200. The mean holds.
- **Wilson** gives the best average accuracy and a worst case of at least 0.91 on [0.05, 0.95], and needs no special functions. That matches the standard advice for small n **[M: Brown, Cai & DasGupta 2001; Newcombe 1998 "seven methods" V: title found via search]**.
- **Clopper-Pearson** is always ≥ 95% but 5–10% wider.
- **Jeffreys** needs the beta quantile (scipy) and dips to 0.885.
- **Bootstrap** of a Bernoulli mean adds nothing over these, and breaks at k = 0 or k = n.
- Wilson's half-width at p = 0.5 is ±13.4 points at n = 50, ±9.6 at 100 and ±6.9 at 200.

```python
from phase3_stats import wilson, mcnemar_exact, newcombe_paired_diff
lo, hi = wilson(k, n)                                    # per checkpoint
d, (dlo, dhi) = newcombe_paired_diff(succ_B, succ_A)     # bool vectors on the SAME seeds, B - A
p = mcnemar_exact(b=int((succ_A & ~succ_B).sum()), c=int((~succ_A & succ_B).sum()))
```

## 2. Regression test between two checkpoints

- **Paired beats unpaired when the outcomes correlate.** On real checkpoints (§4): A vs E, CI width **0.147 paired vs 0.191 unpaired** (Fisher p .36 vs McNemar p .24). For weakly correlated pairs the two are equal (C vs D: 0.178 vs 0.175). Pairing is free in simulation (same seed = same initial state), so always pair.
- **Unpaired tests (Fisher/Barnard, two-proportion z)** are only for comparing results that weren't run on shared seeds, e.g. an imported eval. They need about 1.4× the n for the same MDE (a normal-approximation estimate, **[M]**).
- **Effect size:** Newcombe method 10 (Wilson-based, uses the correlation φ) **[V: Newcombe, Stat Med 17:2635, 1998, search result]**. Tango's score interval is an alternative with similar behavior **[M; Tango 1998, V: letter exists]**. Newcombe is closed form, so use it.
- **MDE (80% power, exact McNemar; `mde_paired`).** "Lose" is the share of A's successes that B fails:

| n paired | p_A = .5, lose 0 | p_A = .5, lose 10% | p_A = .8, lose 10% | **p_A = .555, lose 22% (observed A vs E)** |
|---|---|---|---|---|
| 50 | 15.5 pts | 23.0 | not reachable | 30.5 |
| 100 | 8.0 | 14.5 | not reachable | 19.0 |
| 200 | 4.0 | 9.0 | 10.5 | **12.5** |
| 400 | 2.0 | 6.0 | 7.0 | 8.5 |

  - Honest output format: *"No detectable difference (Δ = −5.0 pts, 95% CI −12.3 to +2.4; McNemar p = .24). At the disagreement observed between these two checkpoints, n = 200 detects improvements of about 12.5 points or more with 80% power."*
- **Sequential/early stopping:** not in phase 3.
  - STEP (Snyder et al., **RSS 2025, arXiv 2503.10966 [V]**) is a sequential test for comparing two IL policies that keeps the type-I error guarantee. It reports up to 32% fewer trials, but it is built for independent (unpaired) trials **[M]**.
  - Wald's SPRT is the classic single-policy version **[M]**.
  - Our rollouts cost about 0.035 s each, so a fixed n = 200 (7 s) beats the complexity. Revisit for real-robot eval, where trials are expensive.
- **Prior art:**
  - Kress-Gazit et al. 2024, "Robot Learning as an Empirical Science" **[V: arXiv 2409.09491]**: report the number of runs, the initial conditions and the success criterion, add statistical tests, and describe failure modes.
  - **`lerobot_eval.py` [V: 0.4.4 source]**:
    - reports `pc_success` as a plain mean with **no CI or test**;
    - counts an episode as a success if it succeeds **at any step** (`reduce(..., "any")`);
    - seeds envs from `start_seed` in batches.
  - Ours is **final-frame** success (the arm must *hold* at the target). The report must state the definition, because the two numbers differ.

## 3. Repeatable rollouts

- **Seeding:**
  - env reset `SeedSequence([eval_base, i])` for episode i (the target is drawn from it);
  - policy RNG = a separate child seed per episode;
  - training `torch.manual_seed` plus a seeded `DataLoader` generator.
  - Store the list of eval seeds in the result JSON. A regression run reuses it.
- **Measured determinism** (same machine): 50 rollouts gave an identical success vector (30/50) and an identical SHA-256 of the final states across 4 processes and torch 1 vs 2 threads. Retraining with the same seed gave identical weight hashes (1 vs 2 threads).
- **Can't promise:**
  - bitwise equality across CPU architectures and BLAS builds (Apple Accelerate vs OpenBLAS/MKL; MuJoCo x86 vs arm64), where borderline episodes can flip;
  - determinism on **MPS/GPU** (nondeterministic kernels; `torch.use_deterministic_algorithms(True)` raises on some ops) **[M]**;
  - determinism for multi-threaded DataLoader workers unless seeded.
  - Rule: evaluate on CPU. Compare checkpoints **on the same machine in the same run**. Treat a cross-machine rerun as a new sample, not a replay.
- **Recommended n and cost:** **200 episodes per checkpoint** (Wilson ±7 pts; paired MDE about 9–13 pts, depending on how often the checkpoints disagree).
  - MLP: 200 rollouts take 6.7 s here (0.034 s/episode, about 2,700 control steps/s).
  - ACT through the adapter: 0.09 s/episode, so 18 s per 200.
  - Mac Air: unmeasured, but expected to be similar per core **[M]**.

## 4. Tiny trained policy: measured A/B and the data → policy link

Setup (`phase3_bc_eval.py`):
- Data: 200 clean episodes plus 100 junk (noise 0.25 ×30, random ×30, hesitation ×20, wrong ×20) from the bench v2 generators, exported with `rrc export` and read with pyarrow.
- Model: MLP 9→256→256→6 predicting `action − state`, Adam at 1e-3, 40 epochs, batch 256.
- Eval: the same 200 seeds for every model. Build and export of 300 episodes took 94 s (one-off).

| Training data | episodes | train | success per train seed (0/1/2, of 200) | mean (seed range) |
|---|---|---|---|---|
| A clean | 200 | 6.7 s | 111 / 89 / 112 | **52.0%** (44.5–56.0) |
| B mixed (unfiltered) | 300 | 10.5 s | 1 / 0 / 1 | **0.3%** (0–0.5) |
| C mixed, **scorer-filtered** (quality ok) | 224 | 7.5 s | 61 / 68 / 38 | 27.8% (19.0–34.0) |
| D mixed, **sim-success-filtered** | 212 | 7.6 s | 51 / 45 / 31 | 21.2% (15.5–25.5) |
| E mixed, **scorer ok AND success** | 196 | 6.9 s | 121 / 110 / 129 | **60.0%** (55.0–64.5) |

(No pooled Wilson interval: 600 rollouts from 3 different checkpoints aren't one binomial, and such an interval would be narrower than the seed spread.)

**Per-junk-type attribution** (QA reran it with the same 200 clean episodes, 3 training seeds each):

| Variant | episodes | success |
|---|---|---|
| clean + 30 **random** | 230 | **0.0%** |
| clean + 30 **noise 0.25** | 230 | **14.2%** |
| clean + 20 **wrong goal** | 220 | **28.2%** |
| clean + 20 **hesitation** | 220 | **57.0%** (≥ clean 52%) |
| all 300, normalization fitted on clean only | 300 | 0.5% (not a normalization artifact) |
| size control: 196 random clean episodes | 196 | 46.0% (E's gain isn't a size effect) |

- **What each filter kept:**
  - The scorer dropped all noise and random episodes and 16/20 hesitation, but **kept all 20 wrong-goal episodes**. That's expected: they move normally.
  - The success filter kept **17 hesitation episodes** (they still succeed) and dropped 8 near-miss clean ones.
  - Only the intersection removed both kinds of junk.
- **Paired tests (train seed 0):**
  - A vs B: +55.0 pts [+47.5, +61.8], p < 1e-4.
  - B → C: +30.0 pts, p < 1e-4.
  - A vs E: −5.0 pts [−12.3, +2.4], **p = .24, no detectable difference**.
  - C vs E: −30.0 pts, p < 1e-4.
- **Product takeaway, measured:**
  - The damage comes from **random-action and high-noise demos** (which the motion scorer removes) and **wrong-goal demos** (which only outcome evidence removes). **Hesitation did no harm** here.
  - So C (scorer only) loses to E because of the 20 wrong-goal episodes, and D (success only) loses because of the 3 surviving noise 0.25 episodes.
  - E beats clean-only on 3/3 seeds because it also drops failed near-miss clean episodes; the size control rules out a size effect. That's a recipe-level result, not significant at the checkpoint level (p = .24).
  - "0.3% unfiltered" is mostly 30 random-action episodes poisoning an MSE regressor, which is specific to this model class.
  - **Caveat:** one task, one model class and 3 training seeds.

## 5. LeRobot policy adapter (lerobot 0.4.4)

```python
cfg = PreTrainedConfig.from_pretrained(path); cfg.device = "cpu"          # path = .../pretrained_model (local)
policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=path)           # saved normalizer/unnormalizer pipelines
policy.reset()                                                            # every episode: clears the action-chunk queue
a = post(policy.select_action(pre({"observation.state": s[None], "observation.environment_state": e[None]})))
```

- **Inputs:** `cfg.input_features` must be a subset of what `so101_reach` provides (`observation.state`, `observation.environment_state`). Image policies are refused with a clear message.
- **Chunking:** `select_action` pops from an internal queue of `n_action_steps`, so `reset()` per episode is mandatory. Temporal ensembling is used if configured.
- **Training ACT state-only** is allowed: ACT's `validate_features` accepts an env state without images, and no vision backbone is built.
  - Command: `python -m lerobot.scripts.lerobot_train --dataset.repo_id=... --dataset.root=<rrc export> --policy.type=act --policy.device=cpu --policy.push_to_hub=false --policy.chunk_size=20 --policy.n_action_steps=20 --policy.dim_model=128 --policy.n_encoder_layers=2 --steps=300 --batch_size=32 --num_workers=0 --eval_freq=0 --wandb.enable=false`
  - Result: **54 s for 300 steps (5.5 steps/s, 1.47 M params)**; 2,000 steps at lr 1e-4 took **5 min 55 s**.
  - `--policy.push_to_hub=false` is required. The default is true.
- **Quality:** 1/100 (300 steps) and **0/100 (2,000 steps)**; median final tip error 13–14 cm.
  - The adapter is correct: mid-episode predicted chunks match the dataset actions within about 0.05 rad (t = 30 and t = 60, checked).
  - The failure is at t = 0. Every episode starts from the same pose, so the first chunk has to infer the whole target→joint mapping, and 3.6 epochs (64k samples) isn't enough. The MLP saw about 700k samples.
  - Getting ACT to useful success likely takes around an hour of CPU **[M, extrapolated]**.
- **Recommendation:** ship the adapter with a *tiny ACT fixture* (300 steps, checked into test data, about 6 MB) as a load/rollout smoke test. Evaluate "real" LeRobot checkpoints that users bring. Hub download is blocked here, so a Hub checkpoint is **untested**.

## 6. R-real-1: `action_tv_ratio` on return-to-home motions

- **Setup:** benchmark v2 (production `bench` generators, including QA's fixed wrong/stall) plus a new **return_home** group: out over U(0.8, 1.2) s, hold U(0.1, 0.4) s, back to the start pose. Scored with the P2-5 numerics (`phase3_tv_fix.py`).
- **The false positive needs a smooth dataset, not just smooth return-home episodes.** The bug only shows when the *whole set* is low-noise, because the flag is relative to the dataset's own median and MAD.
  - Standard noise everywhere (σ ~ U(0.01, 0.04)): return_home is flagged 0/10 even with today's formula.
  - Clean at standard noise with only return_home low-noise (the plan's design): also 0/10 on dev, held-out and private (QA). **This construction does not reproduce DoD 8.**
  - **All groups low-noise** (σ ~ U(0.002, 0.008), closer to a leader arm as in the DoD 8 data): today's formula flags it. **The R-real-1 benchmark needs these whole-set low-noise variants** (plan D3):

| tv denominator | dev return_home flagged (median tv z) | held-out | DoD 4 bars dev (std / low-noise) | held-out |
|---|---|---|---|---|
| net \|a_T − a_0\| (current) | **5/10** (+3.4) | **3/10** (+3.0) | PASS / PASS | PASS / PASS |
| max excursion max_t\|a_t − a_0\| | 0/10 (+0.4) | not run | PASS / PASS | not run |
| **split** \|a_k − a_0\| + \|a_T − a_k\| | **0/10 (−0.8)** | **0/10 (−0.7)** | PASS / PASS | PASS / PASS |

QA reproduced this independently, including on its private set: return_home flags go **net 5/3/2 → split 0/0/0** (dev / held-out / private, all-low noise). Split keeps every DoD 4 bar on all three sets under standard, all-low and mixed noise.

- **Why split rather than max excursion:**
  - Split ≥ net always, so no episode's tv score can rise, apart from a slightly higher floor median.
  - For one-way reaches it equals net.
  - Max excursion also passes but treats an out-and-back as half the displacement.
- **Limit:** multi-waypoint tasks (pick → place → home) still under-count the displacement. A general version would split recursively at the farthest points (a Douglas-Peucker-style simplification). Keep that as a follow-up until real multi-waypoint data shows the need.
- **Watch item:** in the low-noise dev set, clean-only is flagged **5%** (on the bar), by `action_tv_ratio` 3 (short reaches) and `saturation_frac` 2. It's 3% on held-out.

## Open uncertainties

- **Mac:** no timing, determinism or MPS runs here, and all cross-machine claims are **[M]**. Sebi should run `phase3_bc_eval.py` once; it needs about 3 min and lerobot only for the export.
- The BC results cover one task, an MLP only and 3 training seeds. Filtering effects may be smaller with more data or a stronger model.
- ACT usefulness on CPU within minutes isn't shown. Pretrained Hub checkpoints (images, other robots) can't run in a state-only sim; the adapter refuses them.
- The low-noise benchmark variant is my construction to reproduce DoD 8. Real SO-101 noise levels should decide which benchmark noise is canonical.

## Changes applied (Researcher, after QA phase 3 review)

- **R1 (Wilson worst case):** Rec 1 and §1 now say mean ≈ 0.95, worst ≥ 0.91 on [0.05, 0.95] (fine grid, 0.916 at n = 20), and lower near 0 or 1 (0.838 at n = 20).
- **R2 (MDE):** Rec 3, §2 and §3 now say 9–13 pts at n = 200 depending on disagreement. I added an observed-disagreement column to the MDE table (22% lose → 12.5 pts at n = 200, recomputed with `mde_paired`) and reworded the honest-output example.
- **R3 (attribution):**
  - §4 now has QA's per-junk-type table: random and high-noise hurt, wrong goal hurts, hesitation doesn't.
  - The takeaway and Rec 6 are rewritten; E > A is a recipe-level result (3/3 seeds) but not significant at the checkpoint level.
  - §6 now says the R-real-1 benchmark needs whole-set low-noise variants, and that the plan's return_home-only-low-noise design doesn't reproduce DoD 8.
  - The "can only lower scores" wording is limited to the raw ratio.
- **R4 (seed variance):** dropped the pooled Wilson column in favor of per-seed rates plus the range, and changed "about 11" to "10–15 points".

## QA Review (phase 3)

QA, 2026-09-25. **Verdict: APPROVED WITH CHANGES.** The statistics are correct, and every number in §4 and §6 reproduces exactly. What needs changing is how the numbers are explained: the data → policy attribution, the "realistic" MDE and the Wilson worst case are overstated. QA's scripts are in `/tmp/claude-0/-home-claude/98f8b200-727a-5742-bf62-da8620fbd0d2/scratchpad/p3/` (`stats_check.py`, `confound.py`, `e2e_margin.py`, `tvfix_check.py`).

**Statistics, checked against independent references (scipy 1.17.1):**
- Wilson and Clopper-Pearson match `binomtest(...).proportion_ci("wilson" / "exact")` to 2e-16 / 5e-13 (all k, for n ≤ 59, 100, 150 and 200).
- `mcnemar_exact(b, c)` matches `binomtest(b, b + c, .5).pvalue` to 2e-15 (b, c < 80); b = c = 0 gives 1.0.
- Newcombe method 10:
  - identical to QA's own implementation written from the paper (3,000 random tables, max difference 0);
  - **exact-enumeration** coverage over the paired multinomial on a (p_A, p_B, ρ) grid is n = 50: mean 0.947, min 0.930; n = 200: mean 0.949, min 0.947.
- `mde_paired` reproduces every value in the §2 table. Exact and Monte Carlo power at each MDE is 0.80–0.83, and it is the same when only "B better" rejections count.

**Findings:**
1. **The Wilson worst case is a grid artifact.** Coverage computed on a 1e-4 grid over [0.05, 0.95] has minimum **0.916** at n = 20 (0.931 / 0.927 / 0.934 at n = 50 / 100 / 200), not 0.925. Near the edges (p = 0.005–0.05, which is common for us: 0.3%, 100%) it drops to **0.838** at n = 20 and 0.916 at n = 200. The mean (≈ 0.95) holds.
2. **The "realistic" MDE is optimistic.** It assumes B loses 10% of A's successes. The only real checkpoint pair measured (A vs E, seed 0) loses **22%**, so the MDE at n = 200 is **12.5 pts**, not 9–10.
3. **The data → policy attribution in §4 is wrong** (QA reran the §4 exports, 3 training seeds each; success rates below):

   | Variant | Episodes | Success |
   |---|---|---|
   | clean + 30 **random** | 230 | **0.0%** |
   | clean + 30 **noise 0.25** | 230 | **14.2%** |
   | clean + 20 **wrong** | 220 | **28.2%** |
   | clean + 20 **hesitation** | 220 | **57.0%** (≥ clean's 52%) |
   | all 300, normalization from clean only | 300 | 0.5% (so the collapse isn't a normalization artifact) |
   | size control: 196 random clean episodes | 196 | 46.0% (E's 60% isn't a size effect; dropping near-miss failures helps, E > control on all 3 seeds) |

   What this means for the claims:
   - "Junk the scorer can see (hesitation) … costs 25–30 points" is false: hesitation cost nothing here.
   - The C-vs-E gap is the 20 wrong-goal episodes, which only outcome filtering removes.
   - D's drop comes from the 3 surviving noise-0.25 episodes.
   - "60.0% vs 0.3%" is driven mainly by 30 random-action episodes poisoning an MSE regressor, which is specific to this model class.
4. **Confounds, cleared:**
   - The filtered sets differ in size, but not in E's favor: B = 300 is the worst, E = 196, and the size control rules size out.
   - Every variant reuses the same 200 clean episodes (seed 20000).
   - Eval uses `SeedSequence([900000, i])` directly, while training episodes use `SeedSequence([20000 + k, i]).spawn(2)[0]`. The streams are distinct and the targets are continuous, so the eval set is disjoint from training.
5. **The pooled "Wilson 95% (rollouts only)" over 3 training seeds** (e.g. 48.0–56.0) treats 600 rollouts from 3 different checkpoints as one binomial. It's narrower than the seed spread (44.5–56%). Drop it, or report per-seed rates plus the range.
6. **Recipe variance:** the seed-to-seed range is 11 (A), 15 (C), 10 (D) and 9.5 (E) points, so say "10–15 points", not "about 11".
7. **§6 R-real-1:**
   - QA reproduced it independently on dev, held-out and **QA's private set**, with the bars judged by QA's `bench.evaluate`.
   - Split keeps **every DoD 4 bar** on all three sets under standard noise, all-low noise, and "clean standard / return_home low".
   - Return-home flags: **net 5/3/2 → split 0/0/0** (all-low noise).
   - Wording: "split can only lower scores" holds for the raw ratio. z-scores move with the dataset median/MAD.
8. **§6 note:** under the plan's bench design (clean at standard noise, only return_home low-noise), the **unfixed** formula also flags 0/10 on all three sets. That construction doesn't reproduce DoD 8. Only the all-low-noise one does. See the plan's QA review, decision D3.

**Required changes:**
- R1: §1, Rec 1 → "mean ≈ 0.95; worst ≥ 0.91 for p in [0.05, 0.95], n ≥ 20 (fine grid); lower near 0 or 1".
- R2: Rec 3 and §2 → "MDE ≈ 9–13 pts at n = 200 depending on how often the checkpoints disagree (12.5 at the A-vs-E disagreement)". compare prints the MDE at the observed disagreement (plan D2).
- R3: Replace the §4 takeaway and Rec 6 wording with the per-junk-type table above:
  - random and high-noise actions and wrong-goal demos did the damage; hesitation didn't;
  - the scorer removes the first two, outcome evidence removes wrong goals;
  - E > A is a recipe-level result (3/3 seeds) but not a significant checkpoint-level one (p = .24).
- R4: Drop the pooled Wilson column (finding 5), and change "about 11 points" to "10–15 points".

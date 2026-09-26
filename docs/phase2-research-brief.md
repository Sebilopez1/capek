# Research Brief: Phase 2 Dataset Scoring

Researcher, 2026-09-24 (rev 2, after the QA review). Everything below was measured in this container (Linux, CPU, Python 3.11, `/home/claude/.venvs/rrc`) unless marked otherwise.
- Prototype: `docs/spikes/phase2_score_proto.py` (pyarrow + numpy only, the plan's 7 signals, constants in `CFG`). It scores 130 episodes in about 0.3 s.
- Benchmark v2 and evaluation: `docs/spikes/phase2_bench_v2.py` and `phase2_eval_v2.py`.
- Citations are marked **[V]** (checked today via web search/fetch or the installed source) or **[M]** (from memory, unverified).

## Recommendations

1. **The scorer needs no lerobot and no torch.** Read `meta/info.json` and `data/*/*.parquet` directly with pyarrow. That's about 150 MB installed, versus about 8 GB for the lerobot extra on Linux. The same per-frame columns exist in v2.1, so one reader can also cover v2.x read-only. Add `pyarrow` as a `score` extra, or to the base install.
2. **Normalize per joint by a range computed from the data itself.** Use the q01–q99 of each `observation.state` dimension across the dataset, and the data's own per-joint action min/max. Don't use `stats.json`: v2.x has no quantiles **[V: `augment_dataset_quantile_stats.py` exists only under `v30/`]**. Unit invariance check: rad→deg plus a 0–100 gripper, **stored as float32**, gives **identical flags**, with the combined score changing by at most 5e-5 relative (2e-4 absolute).
3. **Score each episode with a per-dataset robust z-score for each signal**, sign set so higher means worse: z = (x − median) / max(1.4826·MAD, floor).
   - Floors (D5): **relative 10% of |median| for ratio-scale signals only** (SPARC arc length, `action_tv_ratio`, `action_hf_energy`, `track_err`). **Absolute floors** for LDLJ (0.5, a log with an arbitrary offset) and the fraction signals (0.03).
   - **Combined score = max z over the 7 signals; flag at z > 3.5.** The "why" is the signals with z > 3 in plain words, with z capped at 10 for display.
   - **Hard flags** stay separate and absolute: NaN/inf, frozen joint, timestamp gap > `1e-4` s, saturation ≥ 20%.
4. **Report two separate axes: *motion quality* and *outcome*** (plan D3/D4). **Motion quality can't detect failed attempts that move normally (early stop, wrong goal, near miss).** Measured: stall is flagged 0/10 on both seed sets, AUROC about 0.56. Only outcome evidence (`next.success`, labels) can show those failures.
5. **Measured on benchmark v2** (plan D1): heterogeneous clean, with every group sharing that timing and noise heterogeneity. Constants were chosen on **dev seeds only**, then run once on held-out seeds.
   - Clean vs. motion junk (noise 0.1, noise 0.25, random, hesitation): **AUROC 1.00 / 1.00, precision 1.00 / 1.00, recall 0.97 / 0.95** (dev / held-out). Clean flagged 0% in the mixed set; clean-only (100 episodes) 2% / 1%. **All D2 bars pass on both seed sets.**
   - Known gaps, measured and reported with no bar: band-limited **wobble is flagged 0/10** (AUROC 0.83 / 0.94). `wrong` was flagged 7–8/10 only through a generator artifact (joint-limit saturation). After the P2-3 fix it is 0/10 (AUROC about 0.5–0.6); smooth wrong goals and uniform junk (B4) are known gaps.
6. **Real Hub data couldn't be checked here.** `huggingface.co` is blocked by the egress policy (403). Use the synthetic "real-like" fixture (§3). **Sebi should download one small real SO-101 dataset on the Mac** (§3). The constants are sim-tuned, so expect to re-check them on real data.
7. The exported-dataset benchmark route (`rrc record` + `rrc export` + lerobot `merge_datasets`) works, but merge drops `rrc_tags.json` (G5), so ground truth has to be kept separately. Benchmark v2 is built in memory through the phase 1 library (QA-owned in the plan). The prototype CLI also ran directly on a merged v3.0 export.

## 1. Per-episode signals (state/action only; plan D5, 7 signals)

Notation:
- `s_t`, `a_t` are normalized per joint by the dataset q01–q99 state range; `fps` comes from `info.json`; `dt = 1/fps`; `T` = frames.
- `net_t = ‖s_{t+w} − s_t‖` with w = 0.5 s (net motion; jitter around a fixed point cancels).

| Signal | Formula (prototype) | Why it indicates junk | Floor / notes |
|---|---|---|---|
| `sparc_state` | SPARC arc length of the **joint-space** speed ‖ṡ‖: normalized FFT magnitude, adaptive cutoff (last frequency with amplitude ≥ 0.05, max 10 Hz), padding 2^(⌈log₂N⌉+4) | Jittery, multi-peaked speed gives a longer arc **[V: Balasubramanian et al., IEEE TBME 2012]**. Used for demo curation on EE speed in "Learning from the Best"/RINSE (+16% success with 1/6 of the data on RoboMimic Transport) **[V: arXiv 2604.23000]**. We use joint space because EE pose needs kinematics | relative |
| `ldlj_state` | −ln( T³/v_peak² · Σ‖s⃛‖² dt ) (velocity form) | Dimensionless jerk **[V: Hogan & Sternad, J Motor Behav 2009]** | **absolute 0.5** (a log, so a relative floor is meaningless) |
| `action_tv_ratio` | Σ‖Δa‖₁ / max(‖a_T − a_0‖₁, **0.5 × dataset median ‖a_T − a_0‖₁**) | Commanded path much longer than the net displacement means dithering. The floored denominator stops short reaches from exploding the ratio | relative |
| `action_hf_energy` | mean(‖Δ²a‖²) | Frame-to-frame chatter; related to the PSD metric **[V: Sojib & Begum, arXiv 2605.01544]** | relative. Depends on fps, so dataset z only |
| `idle_frac` | (# frames **before the last substantial motion** with net_t < max(ε_j, 0.2·p95(net))) / T. Last motion = last t with net_t ≥ max(ε_j, 0.5·p95(net)). ε_j = max(0.02, 2 × dataset jitter floor), where the floor is the p25 across episodes of each episode's p10(net). If p95(net) < ε_j, the value is 1.0 (never moves) | A pause *between* motions (operator hesitation) is a defect. **The terminal hold is excluded**, because that's timing, not quality. Thresholds relative to the episode's own motion keep slow or short reaches from counting as idle. A p25 jitter floor stays stable when a dataset is junk-heavy | **absolute 0.03** |
| `saturation_frac` | fraction of action entries within 1% of the data's per-joint min/max | Hitting joint or teleop limits, clipping, flailing | **absolute 0.03**, plus a hard flag at ≥ 20% |
| `track_err` | mean ‖a_t − s_{t+1}‖₁ | Commands the arm can't follow. **Assumes position control with the same joint order for action and state** (true for SO-100/101 `<joint>.pos`); skip it otherwise | relative |
| hard: NaN, frozen joint, timestamp gap | any non-finite value; ptp(joint) == 0 while that joint varies across the dataset; max \|Δt − dt\| > 1e-4 s | Corrupt recording, dead motor, dropped frames | Absolute. float32 timestamps round by about 2e-7 s (measured) |
| not scored: `length` | robust z of log T | Aborted or overlong attempts | **Not validated** (the sim is fixed at 90 frames); report it as information only |

**Final constants** (`CFG`; chosen on dev seeds only):
- `rel_floor` 0.10; `abs_floor` LDLJ 0.5, idle 0.03, saturation 0.03.
- `tv_den_frac` 0.5; `idle_win_s` 0.5; `idle_eps_mult` 2.0; `idle_r_idle` 0.2; `idle_r_move` 0.5; `flag_z` 3.5.

**Robustness:** in a dev-only sweep (LDLJ floor {0.25, 0.5, 1}, tv {0.25, 0.5, 0.75}, window {0.33, 0.5}, r_idle {0.1, 0.2, 0.3}, fraction floor {0.02, 0.03, 0.05}, multiplier {1.5, 2, 3}), **405 of 486 combinations pass every D2 bar**. The LDLJ floor, tv and fraction floor didn't change pass/fail at all; the failures come from the 0.33 s window and the multiplier of 3. I picked central values rather than the best dev score.

Weaknesses:
- The score is relative, so it assumes most episodes are good; the summary must warn when more than 50% are flagged.
- Scores aren't comparable across datasets.

## 2. Failed attempts: what we can honestly claim

Two separate fields, never merged (plan D4):
- `outcome_sim`: last-frame `next.success`, ground truth only for sims.
- `outcome_label`: `rrc_tags.json` `label` (human); `unlabeled` counts as absent.
- The table shows the label first, then sim, then **"unknown"**, names the source, and marks when label and sim disagree. Teleoperated SO-101 Hub datasets usually have no success column **[M]**. `next.reward` is never used.

The "end state far from the typical final state" heuristic failed (AUROC 0.44, reach targets vary), so we don't use it. Motion signals are worded "looks jerky / hesitates / chatters", never "failed".

The combined motion score predicts sim failure with AUROC only 0.81 on benchmark v2. That correlation is real but it is **not** a failure detector.

Honest claims:
- "Episode X moves unlike the rest of this dataset (why: …)."
- "N episodes are labelled failures."

Not honest:
- "Episode X failed", when there is no outcome evidence.
- Any success rate for real data.

## 3. Reading real SO-100/101 datasets

- **Hub access: blocked here.** `curl https://huggingface.co` gets an HTTP 403 from the egress proxy (a policy denial, so I didn't retry). I couldn't inspect any real dataset.
- **lerobot 0.4.4 (verified in source):**
  - v2.1 raises `BackwardCompatibilityError`, pointing to `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=...`.
  - **v2.0** hits `NotImplementedError` inside that error class, so 0.4.4 can't load or convert it.
  - Older minor versions within v3 only get a warning.
- **Converter gotchas (source):**
  - `--root R` expects the dataset at `R/<repo_id>`.
  - It first tries to download a `v3.0` Hub tag.
  - It converts in place (the original goes to `<name>_old`).
  - **`--push-to-hub` defaults to true**, so always pass `--push-to-hub false`.
  - Local conversion is untested.
- **Recommendation: don't convert.** Read v3.0 (`data/chunk-*/file-*.parquet`) and v2.0/v2.1 (`data/chunk-*/episode_*.parquet`) with our pyarrow reader; both have the same per-frame columns. The v2.1 layout comes from the source's `LEGACY_*` constants plus memory **[M]**; verify it on real data. v2.0 uses the same per-episode parquet layout as v2.1, so the reader handles v2.0 too (plan rev 2). Dispatch on `info.json.codebase_version` and give a clear "unsupported version" error for anything else.
- **Which Hub datasets are v3.0:** unknown here. HF's `lerobot/*` datasets were reportedly converted when v3 launched **[M; the v3 blog/docs exist (V: web search) but the conversion list is unconfirmed]**. Community SO-100 datasets from 2025 are mostly v2.0/v2.1 **[M]**.
- **Feature-name variation** **[M unless noted]**:
  - Joint names vary: `shoulder_pan.pos`… (lerobot so101 follower **[V: source]**), older `main_shoulder_pan`…, or a nested dict of names.
  - Some datasets use `observation.joint_positions`, or add velocity dimensions to the state.
  - Units: degrees or −100..100 for the arm, 0–100 for the gripper (`MotorNormMode` **[V: source]**), handled by data-derived normalization.
  - Require `observation.state` + `action` (overridable with `--state-key` / `--action-key`); skip `track_err` if the dimensions or names differ.
- **Fallback fixture (no network), `tests/fixtures/real_like/`:**
  - (a) rescale to degrees plus a 0–100 gripper, float32, real joint names;
  - (b) truncate 20% of episodes;
  - (c) write a small **v2.1-layout** copy.
- **For Sebi (Mac):** `hf download <dataset> --repo-type dataset --include "meta/*" "data/*" --local-dir ~/rrc-real/<name>` (skips videos). A candidate is `lerobot/svla_so101_pickplace` **[M: name unverified]**.

## 4. Validation: benchmark v2 (measured)

**Benchmark v2** (plan D1, built in memory via the phase 1 library; `phase2_bench_v2.py`):
- **Clean (60):** gravity compensation on, ramp ~U(1.0, 2.5) s, σ ~U(0.01, 0.04) per episode.
- **Ten each, with the same ramp heterogeneity** (σ ~U(0.01, 0.04) unless fixed):
  - nearmiss (no gravity compensation);
  - noise 0.05 / 0.1 / 0.25 (fixed σ);
  - **hesitation** (0.8 s pause at 40% of the ramp, noise continues, then resume; 8–9/10 still succeed);
  - **wobble** (AR(1), φ = 0.9, σ = 0.1);
  - stall; wrong; random.
- Seeds: **dev** mixed base 1000 plus clean-only (100 episodes) base 1500; **held-out** 5000 / 5500. Group k uses seed base+k.

**D2 bars:**

| bar (clean vs. motion junk = noise.1, noise.25, random, hesitation) | dev | held-out |
|---|---|---|
| AUROC ≥ 0.95 | **1.000** | **1.000** |
| precision ≥ 0.90 / recall ≥ 0.85 (z > 3.5) | 1.00 / 0.97 | 1.00 / 0.95 |
| clean flagged in the mixed set ≤ 5% | 0/60 | 0/60 |
| clean-only 100-episode set flagged ≤ 5% | 2% | 1% |
| noise.1 / noise.25 / random flagged ≥ 90% | 10 / 10 / 10 | 10 / 10 / 10 |
| hesitation AUROC ≥ 0.90 (flag rate) | 1.00 (9/10) | 1.00 (8/10) |
| nearmiss flagged ≤ 20% | 0/10 | 0/10 |

**Per group vs. clean, combined max-z** (AUROC, flagged/10), dev | held-out:
- **Motion junk:**
  - noise.1: 1.00, 10 | 1.00, 10
  - noise.25: 1.00, 10 | 1.00, 10
  - random: 1.00, 10 | 1.00, 10
  - hesitation: 1.00, 9 | 1.00, 8
- **Borderline:** noise.05: 0.92, 0 | 1.00, 0. It is ranked above clean but not flagged, which is fine because it's inside the realistic noise range.
- **Outcome-only, reported without a bar:**
  - nearmiss: 0.64, 0 | 0.42, 0
  - **stall: 0.57, 0 | 0.56, 0**
  - wrong: 0.98, 7 | 0.85, 8 (all through `saturation_frac`, the mirrored pose hitting joint limits: a sim artifact). **Superseded by the P2-3 fix:** with an independent wrong goal it's 0.57, 0 | 0.49, 0 (see known gaps)
- **Tracked gap:** **wobble: 0.83, 0 | 0.94, 0.**

**What drives each flag** (top signal of each flagged episode, identical on both seed sets):
- noise and random → `action_hf_energy`;
- hesitation → `idle_frac`;
- wrong → `saturation_frac`.

**Per-signal AUROC, clean vs. motion junk** (dev / held-out):
- sparc .74/.83, ldlj .78/.80, tv_ratio .90/.87, hf_energy .90/.86, idle .72/.70, saturation .82/.80, track_err .90/.85.

No single signal clears 0.95. Max-z works because each junk type has one signal that catches it.

**Junk-majority subset** (10 clean + the other 90): AUROC 1.00 / 1.00, recall 1.00 / 0.95, 0/10 clean flagged, and 46–48% of all episodes flagged. The >50% warning would not quite trigger here. The p25 jitter floor is what kept this stable: with the median, it flagged 5/60 clean.

**Superseded (rev 1):** the rev 1 tables (AUROC 0.98 / recall 0.92 on "mixed3") used 8 signals including the dropped `end_knn`, a homogeneous clean group, and a terminal-hold idle signal. On heterogeneous clean data, that idle signal "caught" stalls only through timing. Rev 1 also claimed "borderline noise 0.05 flagged about 50%"; the real split was noise.05 10/10 and nearmiss 0/10.

**Combining and explaining:**
- Use max-z, not a learned combiner (too few generator types to learn from).
- Row format: `ep 42 | quality: FLAG | outcome: unknown | why: action_hf_energy z=10.0 (2270× median); track_err z=10.0 (40× median)`. Fraction signals are shown as "% of frames".
- A stall or wrong episode with no outcome evidence shows `quality: ok, outcome: unknown` (D3).

## Open uncertainties / known gaps

- **Wobble (band-limited, AR(1) φ = 0.9, σ = 0.1) is not flagged: 0/10 on both seed sets.** It ranks above clean (AUROC 0.83 / 0.94) but stays under z = 3.5. Real operator jitter is closer to band-limited than white, so this is the most likely real-world miss. A candidate fix for later is a band-power signal (1–5 Hz); no bar for phase 2.
- **Smooth failed attempts (stall, wrong goal, near miss) are invisible to motion quality by design.** Outcome evidence is the only route (D3).
- **`wrong` is undetectable by motion (post-fix benchmark, P2-3 `9e8b8c1`).** With the wrong goal drawn from the same distribution as real targets, it is flagged **0/10, AUROC about 0.5–0.6** (0.57 dev / 0.49 held-out). The 7–8/10 in §4 came from a generator artifact: the mirrored target pinned the gripper at its asymmetric limit, which `saturation_frac` caught. That is the expected D3 behavior.
- **B4, uniform junk.** The scores are relative to the dataset. When every episode shares one problem, few or none get flagged. Measured on single-group datasets of 50: noise 0.1 **0/50**, noise 0.25 **0/50**, hesitation **1/50**. A uniformly jittery dataset looks "ok". The summary must say that scores are relative, and dataset-level raw medians (a future absolute reference) are the only route to catching it.
- **Watch item for real data: short-reach false flags ("dithering commands").** `action_tv_ratio` still flags about 1–2% of clean episodes on wide-target data. These are reaches of about **half the median distance** (the one clean-only false flag measured today was at 0.54×), where the floored denominator (0.5 × median displacement) still leaves the ratio high. Early-stopped stalls get flagged by the same path (1/10). Real datasets mixing short and long motions may show more of these; check the flag rate and the "dithering commands" reason on Sebi's first real dataset before trusting it.
- **No real data seen.** Units, name variants, v2.x layout and real teleop noise are unverified. All constants are sim-tuned (dev seeds of one reach task). Pick-place, gripper events, variable lengths and `length` are untested.
- Prior art for later phases: DemInf **[V: Hejna et al., RSS 2025, arXiv 2502.08623]** and CUPID **[V: Agia et al., CoRL 2025, arXiv 2506.19121]**. They're learned (VAEs / a trained policy), so not phase 2. Demo-SCORE **[M]**. There's no built-in LeRobot quality scorer (`lerobot-edit-dataset` only deletes, splits and merges **[V: 0.4.4 scripts]**).

## Changes applied (Researcher, after the QA phase 2 review)

- **R1:**
  - Re-measured everything on benchmark v2 (D1) with dev and held-out seed sets and the plan's exact 7 signals.
  - Normalization is data-derived (q01–q99, action min/max), matching QA's harness.
  - Replaced Recommendation 5 and all of §4. Rev 1 numbers are marked superseded.
- **R2:**
  - Stall, wrong and nearmiss are reclassified as outcome-only. Hesitation was added.
  - `idle_frac` now counts only idle time before the last substantial motion, with thresholds relative to the episode's own motion and a p25 jitter floor.
  - Clean flags: mixed 0%, clean-only 2% / 1%; hesitation AUROC 1.00.
- **R3:**
  - The relative MAD floor applies to ratio-scale signals only.
  - Absolute floors: LDLJ 0.5, fractions 0.03.
  - The `action_tv_ratio` denominator is floored at 0.5 × the dataset median displacement.
  - Constants were confirmed with a dev-only sweep (405/486 pass); held-out was run once afterwards.
- **R4:**
  - The misreported rev 1 numbers are corrected in §4 "Superseded".
  - Unit invariance now reads "flags identical; ≤ 5e-5 relative under float32" (measured; QA saw 2e-6 on its run).
- **R5:** Wobble and smooth failed attempts are listed as known gaps.
- **Prototype API changed:** `score_episodes(eps, fps, cfg) -> raw, Z, comb` replaces `signals`/`IDLE_EPS`/`ABS_FLOOR`, so QA's harness needs a small adapter. `docs/phase2-plan.md` was not touched.

## QA Review (phase 2)

QA, 2026-09-24. **Verdict: CHANGES REQUIRED.** The overall approach holds up:
- pyarrow-only reader;
- data-derived per-joint normalization;
- robust z with max-z;
- hard flags kept separate;
- outcome on its own axis.

The prototype reproduces the brief's table exactly (0.57 s). SPARC is implemented correctly. But the headline numbers still come from a **confounded benchmark**, just a different confound from the one the brief fixed. QA's harness (`/tmp/claude-0/-home-claude/98f8b200-727a-5742-bf62-da8620fbd0d2/scratchpad/qa2/harness.py`) builds benchmarks in memory through the phase 1 library and scores them with the prototype's signal code, using the plan's settings: 7 signals, data q01–q99, data min/max.

**Verified:**
- AUROC table and subset table reproduce to 2 decimals on `mixed3`.
- Held-out seeds (base 5000, 9000) with the **same** recipe: AUROC 1.00, precision 1.00, recall 1.00, 0/60 clean flagged. The plan's 7-signal set does *better* than the brief's 8-signal headline, so the brief's numbers aren't what the plan will build.
- Unit invariance: rad→deg plus a 0–100 gripper, **stored as float32**, leaves flags identical, but the combined score changes by up to **1.5e-3 absolute** (relative 2e-6). The z values reach 1.4e4, so the brief's "1e-15" only holds in float64.
- A clean-only dataset (100 episodes) is flagged at **2–4%**, all by `action_tv_ratio` (see finding 5).

**Findings:**
1. **The clean group is unrealistically homogeneous, and that's a generator confound.** Every clean episode has the same noise σ (0.02) and the same 2.0 s ramp. So the MAD of `idle_frac` and `action_hf_energy` is tiny, and anything generated differently becomes an outlier. Giving clean realistic variation (ramp ~U(1.0, 2.5) s, σ ~U(0.01, 0.04) per episode), seeds 5000:
   - **`stall` flagged 0/10 at z > 3.5** (AUROC still 0.99). A stall at 1 s looks exactly like a fast clean reach followed by a hold. It differs only in *outcome* (it doesn't reach the target). The brief's stall AUROC 1.0 / `idle_frac` 1.0 is a timing artifact, the same class of problem as `wrong` via saturation.
   - `noise005` flagged 10/10 → 1/10. That's fine: it's inside the realistic noise range.
   - Aggregate recall at z > 3.5 drops to **0.75–0.78**, below the plan's 0.85 bar.
2. **Real pauses are missed.** A "hesitation" generator (0.8 s pause mid-reach, then the reach resumes, which is a true motion defect) gets **AUROC 0.43, flagged 0/10**. `idle_frac` counts the terminal hold, which measures timing, not quality. A quick QA variant that only counts idle frames before the last motion catches 10/10 (AUROC 0.95), but flags 5/60 clean, so it needs the Researcher's tuning.
3. **White noise only.** Every noise group is i.i.d. per-frame Gaussian noise, which is exactly what `action_hf_energy` and `track_err` measure. Band-limited wobble (AR(1), φ = 0.9, σ = 0.1) against the heterogeneous clean group is **flagged 0/10** (AUROC 0.82 to 0.91). Real operator jitter is closer to band-limited than to white.
4. **Misreported numbers.**
   - The headline AUROC 0.98 / recall 0.92 uses the 8-signal combination **including `end_knn`**, which the brief itself drops.
   - "Borderline noise 0.05 flagged about 50%" is really `noise005` 10/10 and `nearmiss` 0/10.
   - Recall 0.92 misses only `wrong` episodes, which are a saturation artifact.
5. **Formula issues.**
   - SPARC is correct against Balasubramanian 2015: normalizing by the max equals normalizing by DC for a non-negative speed; the adaptive cutoff is the last frequency above the threshold; frequency is normalized by the cutoff; padding is 2^(⌈log₂N⌉+4). It's computed on joint-space speed, not end-effector speed. That's acceptable, but document it.
   - LDLJ (velocity form): the formula is right, with a negligible T vs T−1 detail. But the **"MAD ≥ 10% of |median|" floor is wrong for LDLJ**, because it's a log with an arbitrary offset. On the clean set its floor is 1.64 while the actual MAD is 0.49, so the floor dominates and down-weights the signal arbitrarily. Apply the relative floor only to ratio-scale signals (`action_tv_ratio`, `action_hf_energy`, `track_err`, SPARC arc length), and give LDLJ an absolute floor.
   - MAD × 1.4826 is correct. The AUROC calculation is correct (ties count 0.5).
   - **`action_tv_ratio` is unstable for short reaches.** Its denominator is `|a_T − a_0|₁ + 1e-3`. The clean-only false positives have a net displacement of 0.59–0.95 against a median of 1.82 (normalized units). Floor the denominator relative to the dataset median displacement.
6. The prototype normalizes with `stats.json` and `action min/max`, while the brief recommends data-derived values. It's noted in the file header, and the data-derived version measured better (point 2 of Verified), so this is fine.

**Required changes:**
- R1: Re-run everything on the heterogeneous **benchmark v2** defined in the plan's QA review (decision D1), on the dev and held-out seed sets, with the plan's exact 7-signal combination. Replace the §4 tables and Recommendation 5.
- R2: Reclassify **stall and wrong as outcome failures** ("motion looks normal; only outcome evidence can show these"). Add **hesitation** (pause then resume) as the motion-defect stand-in, and change the idle signal to count only idle time before the last motion (terminal hold excluded), tuned so clean flags stay ≤ 5%.
- R3: Fix the floors: relative floor for ratio-scale signals only, absolute floor for LDLJ. Floor the `action_tv_ratio` denominator relative to the dataset median displacement.
- R4: Correct the misreported numbers in finding 4, and change "unit invariance 1e-15" to "flags identical; relative 1e-5 under float32".
- R5: Add band-limited wobble to the open uncertainties, as a known gap that is measured and reported without a bar.

### QA Re-review (phase 2)
**APPROVED.**
- **Reproduced independently.** QA used its own benchmark-v2 generator (hesitation pause at a random 25–60% of the ramp rather than the fixed 40%) and its own AUROC and bar code. Only `score_episodes` came from the spike. **Every D2 bar passes** on dev 1000/1500 and held-out 5000/5500, and on **three fresh seed sets nobody had used** (20000/20500, 40000/40500, 71000/71500): AUROC 0.996–1.000, precision 1.00, recall 0.95–1.00, 0/60 clean flagged, clean-only 0–1%.
- **The Researcher's own scripts reproduce the §4 table exactly.** The sweep code evaluates dev only and gives 405/486, matching the brief.
- **Held-out wasn't used for tuning:** the sweep code only ever loads dev, and the untouched fresh sets pass too. One caveat: base 5000 had already appeared in QA's first-round experiments, so that "held-out" set was seen before the redesign. The fresh sets are the clean evidence.
- **Unit invariance under float32:** flags and reasons identical; raw values within 4e-7 relative, z within 5e-6.
- **Stale text to fix, non-blocking:**
  - §3 still says "unsupported v2.0" (the plan now reads v2.0).
  - §2 lists outcome sources as a first-available order (sim, then label). Plan D4 is binding: two separate fields, with the label shown first.

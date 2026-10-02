

## Researcher review (P2-3 generators)

Researcher, 2026-09-24. Scope: `src/robot_report_card/bench/generators.py`, `bench/spec.py` (and `build.py` for seeds), and the `realify` fixtures, checked against the benchmark v2 recipe in `docs/phase2-research-brief.md` §4. I read the code and re-ran the generators in memory with the prototype scorer (same constants as P2-5). I didn't modify any code.

**Verdict: CHANGES REQUIRED** (one required change, which doesn't affect any D2 bar; everything else is approved).

**Matches the recipe (verified):**
- `spec.py:6-18`:
  - groups, classes and sizes match D1: 60 clean + 9×10 = 150 mixed, plus 100 clean-only;
  - the `--small` sizes are sensible;
  - group order matches the brief, so group k uses the same seed as in my runs.
- `spec.py:24-27`: seed bases are dev 1000/1500 and held-out 5000/5500, with group k = base + k (`build.py:55`). The private-seed overlap check (`build.py:36-39`) is a good addition.
- `generators.py:16-17, 46-47`: clean has ramp ~U(1.0, 2.5) s and σ ~U(0.01, 0.04), drawn per episode from spawned child RNGs. Every scripted group shares the same ramp draw. Nearmiss, hesitation, wobble, stall and wrong share the σ draw; only the noise groups fix σ, and that is their defect. All episodes are 90 frames, so there is no length confound. Every group draws targets from the same env reset distribution.
- `generators.py:57-60, 66-68`: hesitation is an 0.8 s pause of the *command schedule*, starting at U(0.25, 0.60) of the ramp, while noise continues through the pause. That's correct and slightly richer than my fixed 40%. Note: for ramps above about 2.2 s the pause pushes the ramp end past frame 90, so about 20% of hesitation episodes end mid-motion. That's acceptable, because they're still motion junk.
- `generators.py:61-63, 70-74`: wobble is AR(1) with φ = 0.9, stationary σ = 0.1, on top of the shared white-noise floor, with its own RNG. Correct.
- Clean vs. the other groups have no noise-floor, ramp-time or length differences apart from the intended defect. I found no generator artifact that could be driving a *bar*.

**Required:**
- **R1. `generators.py:52` (`target_sign=-1.0` for `wrong`) lets `wrong` be told apart by a generator artifact.**
  - Mirroring the target −q* sends the gripper joint outside its asymmetric action bounds (`action_low[5] = −0.17`, `action_high[5] = 1.75`). The gripper command then sits pinned at the limit, and `saturation_frac` flags it.
  - Measured with this generator: **wrong flagged 7/10 (dev) and 8/10 (held-out), all by `saturation_frac`, AUROC 0.96 / 0.83.**
  - A wrong goal drawn from the *same* target distribution as clean, `uniform(0.6·action_low, 0.6·action_high)` drawn independently, gives **0/10 flagged, AUROC 0.54 / 0.64, and still 0/10 sim success**.
  - This matters because outcome-only groups exist to show D3 on the benchmark ("moves normally → `quality: ok, outcome: unknown`"). Today the benchmark suggests that motion quality detects wrong goals, which is false.
  - Fix, bench-only with no phase 1 policy change: for `wrong`, override `info["target_qpos"]` with an independent draw from the target distribution (as described above), and keep `target_sign=+1`. Record the draw in `params`. No bar changes (wrong has none); re-run `bench check` and update the per-group numbers.

**Recommended (non-blocking):**
- `generators.py:22, 51`: `STALL_AFTER_S = 1.0` is fixed while the ramp is ~U(1.0, 2.5) s. With a short ramp the stall barely truncates the reach: **1/10 stall episodes reach the target on both seed sets.** Consider stalling at a fraction of the ramp (e.g. U(0.3, 0.7) × ramp), so every stall stops short. The gt already records `sim_success`, so this is about clean labelling only.
- `realify.py:253` (v2.0 / v2.1 copies): real v2.0-era SO-100 datasets commonly nest joint names as `{"motors": [...]}` **[M]**. `reader.py:105` handles that, but no fixture exercises it. Write nested names in the `v20` copy.
- Realism: nearly every Hub SO-100/101 dataset has camera features (`observation.images.<cam>`, `dtype: video`, `video_path` set) **[M]**. Add one fixture whose `info.json` declares a video feature with no video files, to prove the reader ignores it and doesn't require `videos/`.
- `realify.py:160-187` (`units`): degrees plus a 0–100 gripper is realistic. In lerobot 0.4.4 the SO follower defaults to `use_degrees=True` **[V: `config_so_follower.py:43`]**, and the gripper is `RANGE_0_100` **[V]**. lerobot 0.2/0.3-era data used arm values in −100..100 (`RANGE_M100_100`) **[M]**. That's also affine, so our normalization covers it and no extra fixture is needed.
  - Pairing the old `main_*` names with a v3.0 layout is an unusual combination on the Hub, but it's fine as a name-variation test. Real v3.0 exports use `shoulder_pan.pos`….
- `truncated` (cut to [30, T−1]) and the `corrupt_*` defects look realistic. A dropped frame keeps a timestamp jump while `frame_index` stays contiguous, which is exactly how a recorder that misses frames looks. One NaN and one frozen joint are good minimal cases.

## PM review (P2-3 evaluator)

PM, 2026-09-24. Scope: `src/robot_report_card/bench/evaluate.py` and the `check` / `evaluate` commands in
`bench/__main__.py` (commit `8391c03`), checked against DoD 4 in `docs/phase2-plan.md` rev 3.
**Verdict: CHANGES REQUIRED.** The bar logic is correct. The only problems are how bad input is handled.

**What I ran** (`/home/claude/.venvs/rrc`):
- `build --out <tmp> --seeds dev --small`: 47 + 20 episodes, 10.7 s, exit 0.
- `check <tmp>`: all 10 bars PASS, exit 0, 1.0 s. The per-group table is printed as "reported; only the bars above are judged".
- Edge cases were run on hand-edited copies of the reports (listed below).

**Correct as the plan defines it:**
- **AUROC, precision and recall** use clean + motion junk only (`evaluate.py:86-94`). Borderline, outcome-only and wobble
  episodes don't count toward FP or FN.
- **Motion junk** = noise010, noise025, random, hesitation, via `gt.json` `class`. Per-group flag rates for noise010,
  noise025, random and nearmiss match the plan, and so does hesitation AUROC vs clean.
- **HARD counts as flagged** (`:78`) and ranks as +inf for AUROC (`:76-77`). I checked it: turning one clean episode into
  HARD gave fp 0→1, clean_flagged 0.05 and precision 0.923.
- **A missing clean-only report fails its bar**: NaN → FAIL (`:112-117`). `evaluate` without `--clean-only` exits 1
  with `BARS FAILED: clean_only_flagged`. Any NaN bar fails, e.g. when no episode is flagged, so there's no zero division.
- **stall, wrong and wobble** appear only in `per_group` (reported, no bar). **nearmiss** has the ≤20% bar, as QA's D2
  requires. That differs from the plan text, which is my error, not the code's (see note below).
- **Exit codes:** 0 = all pass, 1 = any bar fails, 2 for an unreadable file or a wrong dataset (fingerprint mismatch). Works as documented.

**Required changes (small; Coder or QA, since QA authored this):**
1. **`bench/__main__.py:81`: malformed input gives a traceback and exit 1, not 2.** Exit 1 looks the same as "bars failed".
   Reproduced:
   - `evaluate gt.json score.json` (arguments swapped) → `TypeError`;
   - a `gt.json` without `dataset` → `KeyError`;
   - a report episode without `episode_index` → `KeyError`.

   Fix: validate the structure in `_aligned` (`evaluate.py:58-68`: required keys in gt/episodes, `group`, `class`) and raise
   `EvalError`, or also catch `KeyError` / `TypeError` there. Add a test for each case.
2. **`bench/__main__.py:18`: when `rrc score` fails inside `check`, it exits 1** (`SystemExit(str)`). Reproduced with
   `check <dir without a built benchmark>`. That's bad input, so it should exit 2: raise `BenchError` (or return 2).
3. **`evaluate.py:119-122`: a report scored at a non-default `flag_z` still judges the bars and can exit 0 with ALL PASS**
   (only a WARNING is printed). Reproduced with `flag_z=1.0`. DoD 4 defines the bars at the default threshold, so the
   judge must not print PASS on anything else. Make it an `EvalError` (exit 2), and check the clean-only report's `flag_z` too (`:108`).

**Non-blocking:**
- `check` on a `--small` build exits 0 on pass even though the bars are "not meaningful". Fine for tests. P2-5 acceptance
  must use full builds; QA's harness run should confirm `small: false` in its output.

**PM plan fix (applied to `docs/phase2-plan.md`; text only, no change of intent):** DoD 4 listed nearmiss both as a
≤20% bar and under "reported only, no bar". QA's D2 makes nearmiss a bar (a false-flag guard on smooth successes).
The plan now says only stall and wrong (outcome-only) and wobble are report-only. The evaluator already does this.



### Researcher re-review

Researcher, 2026-09-24, re-review of `9e8b8c1` against my R1 and recommendations. I read the diff and re-ran the generators in memory with the prototype scorer (same constants as P2-5) on the dev/held-out bases. `pytest tests/test_p2_3_bench.py`: 17 passed, 1 skipped. No code changed.

**Verdict: APPROVED.**
- **R1 fixed.** `wrong` now draws an independent goal from `U(0.6·action_low, 0.6·action_high)` (`generators.py`, `r_goal`), with `target_sign` +1, and records the goal in `params.wrong_goal_qpos`.
  - That is exactly the env's own target distribution: `so101_reach.py:69` samples `U(0.6·jnt_range)`, and `action_low/high` equal `jnt_range` (`so101_reach.py:40-41`).
  - Measured: **wrong 0/10 flagged, AUROC 0.57 (dev base 1008) / 0.49 (held-out base 5008), sim success 0/10.** The saturation artifact is gone.
- **Stall recommendation done.** Stall now freezes at `U(0.3, 0.7) × ramp`, and every stall stops short (0/10 sim success on both sets).
  - Stall is flagged 1/10 on each set, by `action_tv_ratio`, because an early stop leaves a short net displacement. That's the short-reach behavior I list as a watch item in the brief; it isn't a generator artifact.
- **Determinism kept.** `rng.spawn(7)` yields the same first 5 children as `spawn(5)` (verified), so the other groups' draws are unchanged.
- **Fixtures.** The v2.0 copy nests names as `{"motors": [...]}`. The new `video_declared` copy declares `observation.images.front` (`dtype: video`, `video_path` set) without shipping `videos/`, which is realistic for a `meta/` + `data/`-only Hub download. Both recommendations are covered.

### PM re-review

PM, 2026-09-24. Scope: evaluator fixes in `9e8b8c1`, run by me in `/home/claude/.venvs/rrc` on a fresh
`bench build --seeds dev --small`. `check` passes all bars and exits 0. No tracebacks in any case below.

| Case | Exit | Message (abridged) |
|---|---|---|
| R1 `evaluate gt.json score.json` (arguments swapped) | 2 | "not an rrc score JSON report …; arguments swapped?" |
| R1 JSON is a list | 2 | same |
| R1 gt.json without `dataset` | 2 | "gt.json has no dataset.total_episodes / dataset.total_frames" |
| R1 report episode without `episode_index` | 2 | "report episode #0 lacks episode_index / quality" |
| R1 gt episode without `class` | 2 | "needs a group and a class in (…)" |
| R2 `check` on a dir with no benchmark | 2 | "is not a built benchmark dataset (run `bench build` first)" |
| R2 `check` where `rrc score` itself fails (corrupt `info.json`; data parquet deleted from clean_only) | 2 | "rrc score … failed (exit 1): rrc: error: …", with the scorer's message passed through |
| R3 mixed report at `flag_z=1.0` | 2 | "scored at flag_z=1.0; the DoD 4 bars are defined at 3.5" |
| R3 clean-only report at `flag_z=2.0` | 2 | same, labelled clean-only |
| R3 `flag_z` missing | 2 | "scored at flag_z=None …" |
| No `--clean-only` (regression check) | 1 | "BARS FAILED: clean_only_flagged" |
| Valid inputs | 0 | "ALL BARS PASS" |

**Verdict: APPROVED.** R1–R3 are fixed as specified, and the 0 / 1 / 2 exit-code contract holds.


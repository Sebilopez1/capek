# Phase 2 Code Review

QA, 2026-09-24. Scope: 69610f7 (P2-1), 690186e (P2-2), 6a645dc (P2-4), c127b4c (P2-5), 2b0a405 (P2-6), checked against `docs/phase2-plan.md` rev 3.
Environment:
- `/home/claude/.venvs/rrc`;
- a clean `uv` venv with `[dev,score]` only;
- QA's P2-3 harness (8391c03) for the DoD 4 bars.

## Summary

| Subtask | Verdict |
|---|---|
| P2-1 Phase 1 backlog | **APPROVED** |
| P2-2 `rrc record --append` | **APPROVED** |
| P2-4 Reader (v3.0 / v2.1 / v2.0) | **APPROVED** (non-blocking N1) |
| P2-5 Signals + engine | **CHANGES REQUIRED** (R1, R2, R3) |
| P2-6 `rrc score` CLI, JSON, e2e, README | **APPROVED** (README wording is part of R3; STATUS is R4, owner PM) |

The numerics are right and the bars pass on unseen data. The blocking problems are two hard-flag and reason cases that will misfire on real SO-101 data, plus one missing sentence of honesty.

## What I ran

- **`pytest`:** 137 passed and 1 skipped (the gated full benchmark) in 47.5 s. `ruff` is clean.
- **DoD 4 through QA's harness** (`bench check`, reading the `rrc score` JSON): **all 10 bars pass on dev, published held-out and QA's private seeds.**

  | Seed set | AUROC | Precision | Recall | Clean flagged (mixed / clean-only) |
  |---|---|---|---|---|
  | dev | 0.999 | 1.00 | 0.975 | 0 / 1% |
  | held-out | 1.000 | 1.00 | 0.975 | 0 / 0% |
  | private | 1.000 | 1.00 | 0.975 | 0 / 2% |

  Hesitation AUROC is 0.995–1.00. Report-only groups: wobble 0/10, stall 0/10, nearmiss 0/10, wrong 7–8/10 (the joint-limit artifact).
- **DoD 6 on QA's fixtures:**
  - `units` (degrees, 0–100 gripper, `main_*` names, float32) gives identical flags and reasons, with the combined score within 1.3e-5 relative.
  - The v2.1 and v2.0 copies score bit-identically to the source.
  - The truncated copy scores without error.
  - Each corrupt fixture gets exactly its expected hard flag.
- **Demo:**
  - `record` 20 episodes, then `--append` noise 0.25 ×3, random ×2 and stall ×2.
  - `tag` episode 25 as `success` against a sim failure.
  - `export` → `score`: 5/27 flagged (exactly the noise and random episodes, with plain-words reasons). The stalls are `ok`, and `success (label) ≠ sim` is shown.
- **Performance:** reading 13,500 frames takes 0.33 s and scoring takes 0.08 s.
- **Clean `[dev,score]` venv:** 241 MB. `rrc score` works, and no `lerobot`, `torch` or `mujoco` ends up in `sys.modules`.
- **Engine vs. brief:**
  - normalization (state q01–q99, shared scale for action when dims match, action min/max for saturation);
  - robust z (1.4826·MAD, relative floor only for the four ratio-scale signals, absolute LDLJ 0.5 / fractions 0.03);
  - jitter floor p25-of-p10;
  - `tv_ratio` denominator floor;
  - idle-before-last-motion;
  - max-z, flag > 3.5, reasons > 3;
  - outcome fields and precedence.

  All match the brief and the prototype (the Coder's bit-for-bit test passes).

## Break attempts

| # | Input | Result |
|---|---|---|
| B1 | Gripper held still (constant state **and** action) in 8 of 30 otherwise normal episodes, as an SO-101 encoder reads at rest | **8/30 HARD "frozen joint"** (R1) |
| B2 | 3-frame episode | HARD "too short". Correct. |
| B3 | **10-frame episode** (shorter than the 0.5 s idle window) | **FLAG "hesitates: idle 100%"**, a false reason (R2) |
| B4 | Dataset where every episode is noise σ = 0.25 (40 episodes) | **0/40 flagged, no warning**. Median chatter is 0.10 against about 0 for clean data (R3) |
| B5 | v2.1 missing `timestamp` / `frame_index`; v2.0 missing `action`; `v1.6`; v3 info with v2 files | Clean errors naming the column, key or version |
| B6 | `next.success` in only one episode file; one file with `list<double>` state | Clean `rrc: error`, but with an Arrow schema dump as the message (N1) |
| B7 | NaN in action only; 1-episode dataset; 10 identical episodes; info fps ≠ data fps; malformed `rrc_tags.json` | Handled as follows: HARD non-finite (others unaffected); "only 1 episodes" warning; nothing flagged; all HARD timestamp gap with the >50% warning; labels ignored with a note |
| B8 | `--append` with a different `--max-steps`; into a non-session dir; into a file; into a phase 1 session without `max_steps` | All refused cleanly, and the session is byte-identical afterwards. The phase 1 fallback uses its episode length (90), so it refuses 60 and accepts 90 |
| B9 | `--json-out` inside the dataset, as the dataset root, through a symlink into `meta/`, an existing file, a directory; `--json --json-out` inside the dataset; `--threshold nan` | All refused. `--overwrite` works, and missing parent dirs are created |
| B10 | P2-1: `rrc record \| head -1`; a float64 `.npz`; a garbage `.npz` | Exit 141 with finished episodes kept; clean errors, and no `--out` is left behind |

## Rulings on the Coder's deviations

The Coder's numbered list (1–9) wasn't in the repo, the commit messages or STATUS. These are the deviations from plan rev 3 that QA found in the code, including the three the coordinator named.

1. **`--append` creates the session when `--out` is new.** Approved.
2. **`session.json` stores `max_steps`, and older sessions fall back to their common episode length.** Approved, verified in B8.
3. **Hard flag for episodes under 5 frames.** Approved as HARD: in real data, an episode that short is an aborted recording. But the scorable length must also cover the idle window. See **R2**.
4. **Frozen-joint hard flag** (state ptp == 0 while the joint moves elsewhere in the dataset). **Rejected as written (R1).**
   - SO-101 encoders read the same count at rest, and many tasks never use the gripper (or the wrist roll) in some episodes. Such an episode is marked HARD ("corrupt recording"), as B1 shows.
   - A dead motor or stuck encoder looks different: the command moves while the reading stays flat.
5. **>50%-flagged warning.** Keep it, but it's **not a safeguard for the failure it's meant to cover.**
   - Heterogeneous junk is handled well even with 10% clean (recall 0.97, 0 clean flagged, 47% flagged, no warning).
   - The dangerous case is *uniform* junk, such as a jittery leader arm on every episode. It produces **0% flagged** (B4), so no flag-rate rule can catch it.
   - The stronger design is a dataset-level comparison of `signal_medians` against a calibrated reference. It needs real data to calibrate, so it belongs in phase 3, not now. For phase 2, the report must say so explicitly (**R3**).
6. **Warning for fewer than 10 episodes.** Approved.
7. **Episodes with hard flags (other than non-finite or too short) still count in the robust statistics.** Approved: their motion data is valid.
8. **Actions normalized by their own range when the dims differ from the state; `track_err` skipped when dims or names differ.** Approved (matches the brief).
9. **`--json` prints to stdout and writes a file only with `--json-out`. The default `./<name>.rrc_score.json` is refused on a second run without `--overwrite`.** Approved (plan rev 3). See N4 for the UX.

## Required changes

1. **R1 (P2-5, blocker for real data).** `src/robot_report_card/score/engine.py:183` (`hard_flags`).
   - Flag a frozen joint only when the joint's state ptp == 0 **and** its commanded action varies: action ptp > 1% of that joint's dataset action range, and action/state dims correspond.
   - A joint with constant state and constant command is an unused joint: no flag (at most an info note in the JSON).
   - Tests: B1 gives no HARD, and QA's `corrupt_frozen` fixture (state frozen, command moving) stays HARD.
2. **R2 (P2-5).** `engine.py` `MIN_FRAMES` / `signals.idle_before_last_motion`.
   - An episode no longer than the idle window (`round(idle_win_s·fps)` frames) must not get `idle_frac = 1.0`. Either raise the scorable minimum to window + 2 frames (HARD "too short", with the frame count and seconds in the message), or score it with `idle_frac = None`, excluded from z and noted.
   - Test with a 10-frame episode at 30 fps: no "hesitates" reason.
3. **R3 (P2-5/P2-6, honesty).** Extend `RELATIVE_ASSUMPTION` (summary + JSON) and the README paragraph with:

   > If every episode shares the same problem (for example a jittery leader arm), few or none will be flagged.

   - Add B4 to the brief's known gaps.
   - Add a phase 3 backlog item: dataset-level reference check on `signal_medians`.
4. **R4 (PM).** At close-out, STATUS gets:
   - the measured DoD 4 table for dev / held-out / private (above);
   - the rulings on deviations 1–9;
   - the known gaps: wobble, uniform junk, smooth failed attempts.

## Non-blocking

- **N1** reader (`score/reader.py:167`):
  - read `next.success` only when every data file has it (otherwise add a note);
  - replace Arrow's schema dump with "episode files have different columns or types (first differing file: …)".
- **N2** `rrc record --append` with a corrupt `session.json` prints the JSON error without the file name.
- **N3** `--append` doesn't check that the existing episode `.npz` files exist. Export catches it later.
- **N4** README: say that a second `rrc score` run needs `--overwrite` or `--json-out`.
- **N5** Reasons are listed for 3 < z ≤ 3.5 on `ok` rows. That's by design, but consider saying "(below flag threshold)".

## Re-review

QA, 2026-09-24. Scope:
- the Coder's `56b88cd` (P2-fix: R1–R3, N1, N2, N4);
- the bars re-run against it with QA's corrected benchmark (`9e8b8c1`, the P2-3 fix from the Researcher/PM review: independent wrong goal, stall at U(0.3, 0.7) × ramp, evaluator input handling).

**Suite:** 153 passed, 1 skipped (gated) in 47.6 s. `ruff` clean.

**Required changes, re-run:**

| | Before | Now |
|---|---|---|
| R1: gripper held still (flat reading **and** flat command) in 8/30 episodes | 8 HARD "frozen joint" | none flagged |
| R1: dead motor (flat reading, moving command) | HARD | still HARD ("reading never changes although its command moves"); QA's `corrupt_frozen` fixture still HARD |
| R2: 10-frame episode | FLAG "hesitates: idle 100%" | HARD "too short: 10 frames (0.33 s); need at least 17 (0.57 s)"; boundary checked at 16 (HARD) / 17 / 18 (scored), no false idle reason |
| R3: uniform junk (all episodes σ = 0.25) | 0/40 flagged, silent | still 0/40, as expected; the summary, `summary.assumption` in the JSON and the README now say so |
| N1 | Arrow schema dump | "episode files have different columns or types (first differing file: …)"; `next.success` in only some files → ignored with a note |
| N2 / N4 | bare JSON error / undocumented | message names the path; README documents `--overwrite` |

**Final DoD 4 bars** (full builds, `small: false`, `bench check` → `rrc score` JSON; private seeds kept outside the repo):

| Bar | Target | dev | held-out | private |
|---|---|---|---|---|
| Clean vs. motion junk AUROC | ≥ 0.95 | 0.999 | 1.000 | 1.000 |
| Precision (z > 3.5) | ≥ 0.90 | 1.00 | 1.00 | 1.00 |
| Recall (z > 3.5) | ≥ 0.85 | 0.975 | 0.975 | 0.975 |
| Clean flagged, mixed | ≤ 5% | 0/60 | 0/60 | 0/60 |
| Clean-only flagged | ≤ 5% | 1% | 0% | 2% |
| noise 0.1 / 0.25 / random flagged | ≥ 90% each | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 |
| Hesitation AUROC | ≥ 0.90 | 0.995 | 1.000 | 1.000 |
| Nearmiss flagged | ≤ 20% | 0/10 | 0/10 | 0/10 |
| **All bars** | | **PASS** | **PASS** | **PASS** |

**Report-only groups** (flagged/10, AUROC vs. clean; dev | held-out | private):
- borderline noise 0.05: 0, .91 | 0, .98 | 0, .96
- hesitation flag rate: 9 | 9 | 9
- wobble (tracked gap): 0, .82 | 0, .82 | 0, .84
- stall: 0, .61 | 0, .81 | 0, .61 (sim success 0/10 on every set now)
- **wrong (independent goal): 0, .50 | 0, .59 | 0, .62** (sim success 0/10). The old 7–8/10 "detection" was the mirrored-target joint-limit artifact. Motion quality now behaves as D3 says: a normal-looking reach to the wrong place is `ok`, and only outcome evidence can show it.

**New observation (non-blocking):** on a 200-episode synthetic clean set with a wider target spread than the benchmark, 4/200 (2.0%) are flagged. All of them are "dithering commands" on short reaches whose net displacement is 0.45–0.55× the median, right at the `action_tv_ratio` denominator floor. That's inside the ≤ 5% bar, but it's the most likely false-flag source on real data, especially tasks that start and end at home. Add it to the watch list for Sebi's DoD 8 run.

### Final verdicts

| Subtask | Verdict |
|---|---|
| P2-1 Phase 1 backlog | **APPROVED** |
| P2-2 `rrc record --append` | **APPROVED** |
| P2-3 Benchmark, evaluator, fixtures (QA-authored) | Fixes in `9e8b8c1`; **awaiting Researcher/PM re-review** |
| P2-4 Reader | **APPROVED** |
| P2-5 Signals + engine | **APPROVED** |
| P2-6 `rrc score` CLI, JSON, e2e, README | **APPROVED** |

Still open: R4 (PM). STATUS needs the table above, the deviation rulings, the known gaps (wobble, uniform junk, smooth failed attempts) and the `tv_ratio` short-reach note. After that: DoD 8 (Sebi's real-data run).

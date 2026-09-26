# Phase 2 Plan — Dataset Scoring

Author: PM · Date: 2026-09-24 · Status: rev 3, QA approved (rev 2) + re-review text corrections applied
Inputs: `docs/phase2-research-brief.md` (signals, benchmark and AUROC numbers), `docs/spikes/phase2_score_proto.py`,
`STATUS.md`. Sebi's scope decisions: score **any LeRobot dataset on disk** (our sim exports and real SO-100/101
Hub data), validate mainly on sim ground truth, output a **terminal table + JSON** (no HTML yet).

## Goal

`rrc score <dataset>` tells you which episodes look like junk and why, in plain words, without needing lerobot
or torch. It keeps **motion quality** (what the joint/action traces look like) separate from **outcome**
(did it succeed). Outcome comes only from evidence in the dataset, never from motion.
Out of scope: HTML report, cameras/video signals, learned scorers (DemInf/CUPID), policy eval, Hub download
inside `rrc`, editing or deleting episodes.

## Definition of Done

1. `pip install -e ".[dev,score]"` (no `sim`, no `lerobot`) is enough to run `rrc score`. A test checks
   `lerobot` and `torch` are not in `sys.modules` after scoring.
2. `rrc score DATASET` prints one row per episode:
   `ep | frames | quality (ok/FLAG/HARD) | score | outcome (+source) | why`.
   It also prints a dataset summary: % flagged, top reasons, a warning when >50% are flagged ("scores assume most
   episodes are good"), and the **honesty statement** (D3): *"Motion quality can't detect failed attempts that move
   normally (early stop, wrong goal). Only outcome evidence (success column or labels) can."* It writes a JSON report
   (`--json-out`, default `./<dataset-dir-name>.rrc_score.json`). An existing file is refused unless `--overwrite`.
   **It never writes into the dataset.**
3. It reads **v3.0**, **v2.1** and **v2.0** layouts (dispatch on `meta/info.json` `codebase_version`; v2.0/v2.1 share
   the per-episode parquet layout). v3.0 and v2.x copies of the same data give identical scores. Other versions →
   a clean "unsupported version" error, no traceback.
4. **Bars on benchmark v2 (P2-3), checked by QA's harness against `gt.json`, and they must pass on BOTH the dev and
   held-out seed sets.** Thresholds are tuned on dev seeds only.
   - Clean vs motion junk (noise 0.1, noise 0.25, random, hesitation): AUROC ≥ **0.95**; at the default threshold
     precision ≥ **0.90**, recall ≥ **0.85**.
   - Clean flagged ≤ **5%** in the mixed set, and a **100-episode clean-only** dataset flagged ≤ **5%**.
   - Noise 0.1, noise 0.25 and random each flagged ≥ **90%**. Hesitation AUROC ≥ **0.90** (flag rate reported).
     Nearmiss flagged ≤ **20%**.
   - Reported only, no bar: outcome-only stall and wrong, and wobble (tracked gap). Nearmiss is outcome-only but keeps
     the ≤20% false-flag bar above. A missed bar is
     reported by the Coder, not hidden by retuning.
5. **Outcome honesty (D4):** JSON has separate `outcome_sim` (last-frame `next.success`) and `outcome_label`
   (`rrc_tags.json`; `unlabeled` counts as absent). The table shows label, else sim, else `unknown`, with the source,
   and marks disagreement (`fail (label) ≠ sim`). A stall/wrong episode with no outcome evidence shows
   `quality: ok, outcome: unknown` (tested). No motion reason contains "fail"/"success" (grep test).
6. **Unit invariance:** real-like variant (a) (degrees, 0–100 gripper, renamed joints, float32) gives **identical flags
   and reasons**, with raw signals and z within **relative 1e-4** of the source; variant (b) (variable lengths)
   scores without error.
7. `pytest` passes offline in < 3 min. Scoring a 150-episode / 13,500-frame benchmark takes < 5 s. STATUS has the measured AUROC table.
8. **Real data, macOS (Sebi):** on his Mac, Sebi downloads one real SO-100/101 dataset (`meta/` + `data/` only),
   runs `rrc score`, and sends `info.json` + the JSON report. It must run without error, or fail with a clear message.
   The Researcher reviews the output. Thresholds are *not* retuned on it in phase 2; findings go to STATUS.

## Design decisions

- **Reader:** pyarrow + numpy only (new `score` extra: `pyarrow>=15`). Reads `data/**/*.parquet` directly (v3.0
  `file-*`, v2.0/v2.1 `episode_*`), sorts by (`episode_index`, `frame_index`), columnar numpy conversion. Keys default to
  `observation.state` / `action` (`--state-key` / `--action-key`); `track_err` is skipped, with a note, if dims differ.
- **Normalization:** per joint, by the data's own q01–q99 range (not `stats.json`, which v2.x lacks).
- **Signals (brief §1):** motion = `sparc_state`, `ldlj_state`, `action_tv_ratio`, `action_hf_energy`, `idle_frac`
  (adaptive ε), `saturation_frac`, `track_err`. Hard flags (absolute): NaN/inf, frozen joint, timestamp gap > 1e-4 s,
  saturation ≥ 20%. `end_knn` dropped. `length` reported but **not** in the combined score (unvalidated: sim is fixed-length).
  Fixes (D5): `idle_frac` counts only idle time *before the last motion* (the terminal hold is excluded);
  the `action_tv_ratio` denominator is floored relative to the dataset's median net displacement. The Researcher
  confirms these constants on dev seeds before P2-5 closes.
- **Score:** per-dataset robust z (median/MAD), higher = worse. The relative MAD floor (10% of |median|) applies only to
  ratio-scale signals; fractions get 0.03 and LDLJ an absolute floor.
  Combined = max z over motion signals, flag at z > 3.5 (`--threshold`), displayed z capped at 10. "Why" = signals
  with z > 3 in plain words with ×median ratios ("looks jittery: action chatter 35× dataset median"); never "failed".
- **Outcome (D4):** two fields, never merged: `outcome_sim` and `outcome_label`, as in DoD 5. `next.reward` is never used.
- **JSON report** (`schema_version: 1`): tool/version, dataset path, `codebase_version`, fps, keys used, thresholds,
  skipped signals + reasons, summary + honesty statement; per episode raw signals, z, combined, quality, hard flags,
  reasons, `outcome_sim`, `outcome_label`, `outcome_disagree`, length. Field names freeze at QA approval (phase 3 and the hosted tier will read it).

## Subtasks (ordered)

### P2-1 Phase 1 backlog cleanup
- **Scope:** `rrc export` catches `ValueError` from a corrupt `.npz`; `rrc record | head` exits like SIGPIPE (141,
  session valid) instead of "N episodes saved" + exit 1.
- **Acceptance:** regression tests for both; the full phase 1 suite stays green.
- **Depends on:** nothing. **Does:** Coder. **Reviews:** QA.

### P2-2 `rrc record --append`
- **Scope:** `--append` adds episodes to an existing session; refused unless env, fps, max_steps and feature spec
  match `session.json`. Indices continue; episode `i` still uses `SeedSequence([seed, i])`. Without `--append` the
  non-empty-dir refusal is unchanged.
- **Acceptance:** three appends with different policies/noise → one session, contiguous indices; export carries each
  episode's policy/params into `rrc_tags.json`; mismatched `--max-steps` refused, session untouched; interrupted
  append leaves the session valid; re-running the same commands gives identical arrays.
- **Depends on:** P2-1 (same files). **Does:** Coder. **Reviews:** QA.

### P2-3 Benchmark v2 builder, QA harness, real-like fixtures (test data, kept independent of the scorer)
- **Scope:** `python -m robot_report_card.bench build --out DIR --seeds dev|heldout [--small]`, one declarative table
  (D1). **Clean (60):** scripted, gravity comp on, per-episode ramp ~U(1.0, 2.5) s, noise σ ~U(0.01, 0.04). **10
  each, same timing spread:** nearmiss (no gravity comp), noise 0.05/0.1/0.25, hesitation (0.8 s pause mid-ramp, then
  resume), wobble (AR(1), φ 0.9, σ 0.1), stall, wrong, random. New generators use the library API, with no new
  `rrc record` flags. Classes: motion junk = {noise .1, noise .25, random, hesitation}, borderline = {noise .05},
  outcome-only = {stall, wrong, nearmiss}, tracked gap = {wobble}. Also a 100-episode clean-only set.
  Records into one session via `--append`, exports once, and writes `gt.json` `{export idx: group, class, policy,
  params, sim_success}` cross-checked against `rrc_tags.json`. **QA harness** `bench evaluate <score.json> <gt.json>`
  computes every DoD 4 bar. It is the only code that judges the bars, and the scorer never sees `gt.json`. `bench realify` makes
  (a) degrees + 0–100 gripper + renamed joints, (b) 20% of episodes truncated, (c) v2.1- and v2.0-layout copies (one parquet
  per episode, matching `info.json`) via pyarrow; plus tiny corrupt fixtures (NaN, frozen joint, dropped frame).
- **Acceptance:** 150 episodes / 13,500 frames per mixed seed set, deterministic, dev and held-out seeds disjoint, build < 60 s each;
  `gt.json` agrees with `rrc_tags.json`; the harness reproduces AUROC on a hand-checked toy input; session-scoped test
  fixtures; export-dependent tests use `importorskip("lerobot")`. Held-out datasets are not shown to the Coder until P2-5 is submitted.
- **Depends on:** P2-2 (lerobot extra for export). **Does:** QA, so ground truth isn't written by the scorer's author.
  **Reviews:** Researcher (recipe matches the brief and has no noise-floor confound).

### P2-4 Dataset reader (v3.0 + v2.0/v2.1, no lerobot)
- **Scope:** `robot_report_card/score/reader.py`: read `info.json`, dispatch on version, detect keys, return
  per-episode state/action/timestamp/optional `next.success` plus `rrc_tags.json`; clear errors for unknown versions,
  missing `info.json`, missing keys, mismatched dims.
- **Acceptance:** arrays equal `LeRobotDataset`'s on the benchmark (lerobot venv); v2.1 and v2.0 copies identical; import guard
  passes; 13,500 frames read in < 1 s.
- **Depends on:** P2-3 fixtures (development can start on a tiny synthetic parquet). **Does:** Coder. **Reviews:** Researcher (layouts), QA.

### P2-5 Signals + scoring engine
- **Scope:** `robot_report_card/score/signals.py` and `engine.py`. This is production code for the brief's signals with the D5 fixes, normalization,
  robust z, hard flags, combined score, reason text and the two outcome fields. It is ported from the spike, not imported.
- **Acceptance:** QA's harness passes DoD 4 on **dev and held-out** seeds, and on a **private QA held-out seed set**
  (not written in any doc; the published held-out set is a second check); DoD 5–6 pass as tests; each corrupt
  fixture gets the right hard flag; unit tests per signal on hand-built traces (sine vs noisy sine, stall plateau,
  pause-then-resume, terminal hold not idle, saturated actions). The measured table for both seed sets, including the
  report-only groups, goes in STATUS.
- **Depends on:** P2-3, P2-4. **Does:** Coder. **Reviews:** Researcher (formulas, D5 constants on dev seeds), QA (runs the held-out bars).

### P2-6 `rrc score` CLI, JSON report, e2e, docs
- **Scope:** `score` subcommand (`--json-out`, `--overwrite`, `--threshold`, `--state-key`, `--action-key`,
  `--only-flagged`, `--json`); JSON schema v1; README "Score a dataset" with the D3 honesty statement and the Mac recipe
  `hf download <repo> --repo-type dataset --include "meta/*" "data/*" --local-dir ~/rrc-real/<name>`.
- **Acceptance:** e2e via the installed `rrc`: `record --append` → `export` → `score`; checks table, JSON schema and
  the >50%-flagged warning (junk-majority subset), the honesty statement, and that an existing `--json-out` is refused
  without `--overwrite`. QA runs the README steps in a clean `[dev,score]` venv; piping behaves.
- **Depends on:** P2-5. **Does:** Coder (CLI), QA (e2e), PM (README/STATUS). **Reviews:** QA reviews the Coder; Coder reviews QA's e2e; QA reviews the PM's docs.

After P2-6: Sebi runs DoD 8 on his Mac, and the Researcher reviews the output before the phase 2 milestone report.

## Risks

| Risk | Mitigation |
|---|---|
| Thresholds (3.5, ε floor, 0.5 s window) are tuned only on sim reach and may over- or under-flag on real teleop | Heterogeneous clean group + held-out seeds; clean-only ≤5% bar; DoD 8 on real data; `--threshold` exposed |
| Bars pass while the flag misses real junk (v1 benchmark was confounded: stall/hesitation/wobble flagged 0/10) | Benchmark v2 (D1); outcome-only and wobble reported, not hidden; QA's harness judges |
| The relative score assumes most episodes are good; junk-majority data may be mis-scored (v2 measures AUROC 1.00, recall 0.95–1.00, but 46–48% flagged, so the warning may not trigger) | Keep the >50%-flagged summary warning; state the assumption in README and JSON |
| The v2.1 layout and real key/name variants come from memory (Hub blocked here) | Our own v2.1 fixture, `--state-key` / `--action-key`, clear errors; Sebi's real-data run is the check |
| `wrong` is caught only via joint-limit saturation (a sim artifact); smooth wrong-goal motion is undetectable | Don't claim it. Outcome only from evidence; documented as a limit |
| Benchmark confounds (fixed timing / noise floor fake high AUROC) | Randomized ramp and noise in every group; Researcher reviews the recipe; QA owns ground truth |
| lerobot `merge_datasets` drops `rrc_tags.json` | Not used: the benchmark is one appended session exported once |
| Test time grows with two 150-episode benchmarks + clean-only set | Session-scoped fixture; `--small` variant for fast unit tests |

## Defaults (Sebi can override at milestone)

1. `pyarrow` ships as a `score` extra, not a core dependency, so record/tag stay light.
2. `rrc score` never modifies the dataset. The JSON goes next to where you run it, not into `<dataset>/meta/`.
3. `length` is reported but not part of the combined score until it's validated on variable-length data.

## Needs Sebi's call

1. **Real-data check (DoD 8):** after P2-6, run `rrc score` on one real SO-100/101 dataset on your Mac and send
   `info.json` + the JSON report. If you've recorded your own SO-101 data, prefer that. Otherwise pick a small
   Hub SO-101 dataset (the brief suggests `lerobot/svla_so101_pickplace`, name unverified).

## QA Review (phase 2)

QA, 2026-09-24. **Verdict: CHANGES REQUIRED.**
- The shape of the plan is right and fits one session: pyarrow reader, data-derived normalization, robust max-z, hard flags, outcome on its own axis, QA-owned ground truth, no writes into the dataset.
- The **DoD 4 bars are set just under the prototype's numbers, on the same confounded benchmark and the same seeds.** That makes them a regression test of the spike, not a validation. Details and measurements are in the brief's QA review. These requirements come from them.

**Findings:**
1. With a realistic clean group (per-episode ramp 1–2.5 s, noise σ 0.01–0.04):
   - **stall is flagged 0/10** (AUROC 0.99, so it ranks fine but isn't flagged);
   - a genuine pause-then-resume ("hesitation") is flagged 0/10;
   - band-limited wobble is flagged 0/10;
   - aggregate recall is 0.75, below the 0.85 bar.

   The current bars would pass while the flag misses these cases.
2. **DoD 4 counts `wrong` in "hard junk"** for the aggregate AUROC and recall, and asks for stall ≥ 0.95. `wrong` is caught only through the joint-limit artifact, and stall only through fixed timing. Both are outcome failures that motion can't see. Leaving them in contradicts the plan's own honesty rule.
3. **DoD 6 "within 1e-6" fails** when the real-like variant is stored as float32: combined scores differ by up to 1.5e-3 because z reaches 1.4e4. The flags are identical.
4. **No bar covers a clean-only dataset**, and that's the most common real case (a user's own mostly-good demos). Measured: 2–4% flagged, all by `action_tv_ratio` on short reaches.
5. **Outcome precedence** puts `next.success` before the human `label`. Phase 1 kept `sim_success` and `label` separate and never merged them. The plan should do the same here: show both, and don't silently pick one.
6. **Plan vs. brief is consistent** apart from the brief's stale numbers. The plan's calls stand:
   - `--append` instead of `merge_datasets`;
   - no `next.reward` for outcome;
   - `length` kept out of the combined score;
   - `end_knn` dropped;
   - `pyarrow` as a `score` extra.
7. **Scope fits one session** provided three things hold:
   - the benchmark v2 generators (heterogeneous clean, hesitation, wobble) live in the QA-owned bench builder, using the library API (no new `rrc record` flags);
   - the signal fixes stay small (R3 in the brief's review);
   - thresholds are **not** re-tuned on the held-out seeds.
8. Non-blocking:
   - v2.0 has the same per-frame parquet layout as v2.1. Consider reading it rather than refusing it, since Sebi's real pick may be v2.0.
   - The default `--json-out` silently overwrites an existing file.

**Required changes:**
- R1: Rewrite DoD 4 per decisions D1–D3 below.
- R2: DoD 6 becomes: identical flags and reasons; raw signals and z within relative 1e-4.
- R3: DoD 5 and the design use the outcome rule in D4.
- R4: P2-3 builds benchmark v2 (D1) with dev and held-out seed sets. P2-5 acceptance runs the bars on both, using QA's harness reading `gt.json`, not the scorer's own evaluation code.
- R5: The report and README must state the limitation in D3.

**Reconciled decisions (binding for the Coder):**
- **D1 Benchmark v2 (P2-3, QA-owned):**
  - **Clean (60):** scripted, gravity compensation on, per-episode ramp ~U(1.0, 2.5) s, noise σ ~U(0.01, 0.04).
  - **Other groups (10 each), same timing heterogeneity:** nearmiss (no gravity compensation), noise 0.05 / 0.1 / 0.25, **hesitation** (a 0.8 s pause mid-ramp, then resume), **wobble** (AR(1), φ = 0.9, σ = 0.1), stall, wrong, random.
  - **Classes:** motion junk = {noise 0.1, noise 0.25, random, hesitation}. Borderline = {noise 0.05}. Outcome-only = {stall, wrong, nearmiss}. Tracked gap = {wobble}.
  - **Seeds:** a dev seed set and a held-out seed set.
- **D2 Bars (must pass on both seed sets):**
  - clean vs. motion junk: AUROC ≥ 0.95, precision ≥ 0.90, recall ≥ 0.85;
  - clean flagged ≤ 5% in the mixed set;
  - **a 100-episode clean-only dataset flagged ≤ 5%**;
  - noise 0.1, noise 0.25 and random each flagged ≥ 90%;
  - hesitation AUROC ≥ 0.90, with the flag rate reported;
  - nearmiss flagged ≤ 20%.

  Outcome-only groups and wobble are measured and reported only (no bar). If a bar misses, the Coder reports it. Thresholds are tuned on dev seeds only, never on held-out.
- **D3 Honesty text:** the summary and README say *"Motion quality can't detect failed attempts that move normally (early stop, wrong goal). Only outcome evidence (success column or labels) can."* A stall or wrong episode with no outcome evidence must show `quality: ok, outcome: unknown`. The tests assert this.
- **D4 Outcome:** the JSON carries `outcome_sim` (last-frame `next.success`) and `outcome_label` (from `rrc_tags.json`; `unlabeled` counts as absent) separately. The table shows `label` if present, else `sim`, else `unknown`, with the source, and marks a disagreement between the two (e.g. `fail (label) ≠ sim`). `next.reward` is never used.
- **D5 Signals:** the plan's 7 signals, with these changes:
  - `idle_frac` counts only idle time before the last motion (terminal hold excluded);
  - `action_tv_ratio` denominator floored relative to the dataset median displacement;
  - the MAD relative floor applies to ratio-scale signals only, with an absolute floor for LDLJ.

  The Researcher confirms the constants on dev seeds before P2-5 is closed. Everything else follows the plan as written.

### Changes applied (PM, 2026-09-24, rev 2)
- R1/D1/D2: DoD 4 rewritten. The bars are measured on benchmark v2 (heterogeneous clean, hesitation, wobble), must pass on
  dev and held-out seeds, and are judged by QA's `bench evaluate` harness against `gt.json`. Outcome-only groups and wobble are reported only, and there's a clean-only ≤5% bar.
- R2: DoD 6 is now identical flags and reasons, with values within relative 1e-4.
- R3/D4: separate `outcome_sim` / `outcome_label`; disagreement is marked; stall/wrong without evidence show `ok` / `unknown`.
- R4: P2-3 builds benchmark v2 with dev and held-out seed sets plus the harness; P2-5 acceptance runs on both sets.
- R5/D3: the honesty statement goes in the report summary, the JSON and the README (P2-6).
- D5: signal fixes (idle before last motion, `action_tv_ratio` floor, MAD floors per signal type), with constants confirmed by the Researcher on dev seeds.
- Non-blocking suggestions adopted: v2.0 is read (same layout as v2.1); `--json-out` refuses to overwrite without `--overwrite`.

### QA Re-review (phase 2)
**APPROVED (rev 2).**
- **Bars:** QA reproduced every DoD 4 bar with its own generator and evaluation code, on dev, on held-out, and on three fresh seed sets (details in the brief's re-review).
- **Plan vs. brief:** they agree on signals, D5 fixes, bars, classes and honesty.
- **Four text corrections, to make before P2-3 starts (not blockers):**
  1. The fraction MAD floor is **0.03** (brief `CFG`), not 0.02.
  2. Benchmark v2 is **150** episodes per mixed set (60 + 9×10) / 13,500 frames, not 160 / 11,700. Fix DoD 7 and the P2-3 and P2-4 acceptance numbers.
  3. The junk-majority risk row cites the v1 numbers (0.91 / 0.72). v2 measures AUROC 1.00 and recall 0.95–1.00 with 46–48% flagged, so the >50% warning may not trigger. Keep the warning and update the numbers.
  4. The held-out base 5000/5500 was already used in QA's first-round experiments. **For P2-5 acceptance, QA uses a private held-out seed base that isn't written in any doc.** The published 5000/5500 stays as a second check.

# STATUS

Crew log. Append a new session entry at the top after every work session.
Nothing is "Shipped" until a different role than the author has reviewed it.

**Current phase:** 4 — Free Release (release-ready; Sebi publishes)
**Product-direction lock:** Sep 11 – Oct 23, 2026 (don't relitigate)

---

## Session 4 — 2026-09-25

### Shipped (QA: all P4 code approved, rehearsal PASS; `docs/reviews/phase4-code-review.md` → Re-review, `58660ee`)
- `20e6606` P4-0: phase 4 plan rev 2 + brief (QA reviewed); third-party dataset name redacted in tracked files
- `6dd71de` P4-1: PEP 639 packaging (0.1.0, Apache-2.0, 3 license files), PyPI-form install hints, owner placeholder
  + `tools/set_github_owner.py`, `tools/release_check.py` (never uploads)
- `7ceaaf5` P4-2: CI (light matrix ubuntu/macOS × py3.10–3.12, heavy CPU-torch job, weekly full bench) and the
  trusted-publishing `release.yml` (`pypi` / `testpypi` environments)
- `eb0254a` P4-3: `rrc record --mix`, `--list-groups`, independent-goal `--policy wrong` (`--wrong-goal mirrored`)
- `2e01a7e` P4-4: `rrc report` (Data, Policy, Regression, Verdict, Can't tell; terminal / Markdown / HTML / JSON)
- `8dcb33d` P4-5: report-card rules test (QA, test-first; 26/26, every rule mutation-caught)
- `6d2f350` P4-6: N3 (`--append` checks existing episode files); R-real-3 deferred
- `b102d54` P4-7/8: README rewrite for PyPI, CHANGELOG, CONTRIBUTING, SECURITY, RELEASE.md, launch drafts (PM)
- `b355a98` P4-fix: QA R1 (linking respects the training filter), R2 (release fails on the placeholder), R3 (no denylist leaks)
- `58660ee` QA re-review; PM doc fixes D1–D4 in this close-out commit
- Reviews: `docs/reviews/phase4-code-review.md`

### Measured
- `pytest`: **306 passed, 3 skipped** in about 142 s (budget 150 s). `ruff` check + format clean.
- **release_check (QA rehearsal, fresh clone, temp owner): 9/9 PASS in 38 s.** Build, `twine check` and metadata
  (2.5, Apache-2.0, 3 license files) pass. The wheel is **6.1 MB**. The base wheel install is **74 MB**, and the
  quickstart `[score,sim]` 325 MB. The sdist light suite gives 239 passed / 29 skipped. Placeholder: none left.
  Redactions: 0 matches in 142 files, but **2 history commits** still contain a denylist entry.
- In the crew repo (owner not set), release_check passes everything except the placeholder, as expected.
- The README "Evaluate a policy" block, run verbatim from the built wheel (clean py3.11 venv), took 46 s: 0/200 vs
  29/200, Δ +14.5 [+9.9, +20.0], p = 3.7e-9. That matches phase 3 and the `bench record` weight hashes.

### Deviations and rulings
1. `rrc report`: Inconclusive lists R5's unmet conditions; `--json-out` defaults to `./rrc_report.json` with
   `--overwrite`; inline dark-mode CSS; linking is computed only for the compare's A/B sides. All approved.
2. Linking ignored the training filter. **Rejected → fixed (R1).** Only `--keep all` checkpoints are "trained on" the
   scored dataset. A filtered checkpoint is reported as a subset (k of N episodes, x% failed) and never satisfies R5.
3. The release must fail on the placeholder (R2): `set_github_owner.py --check` exits 1, and `release.yml` runs it in the build job.
4. `release_check.py` prints no denylist text, only entry numbers and locations. It has no crew-machine default path (R3).
5. **R-real-3 stays deferred.** QA's movement-duration LDLJ variant changed 0 flags on every seed set, because on our
   data the moving span is almost always the whole episode. It neither reproduces nor fixes the short-episode bias, and
   landing it would break parity with the phase 2 prototype for no gain.
6. `bench record` stays as a thin alias of `rrc record --mix`. The group names are a public API from 0.1.0.

### Known gaps
- **CI and release workflows have never run on GitHub.** They were written and linted offline; Sebi's first push is the real test.
- Untested here (proxy): the TestPyPI upload, the CPU-only torch install and the macOS CI job. The PyPI name is unconfirmed
  until upload (the JSON API showed 404).
- R-real-3 (LDLJ duration bias) is deferred (see ruling 5).
- The phase 3 Mac check is still pending.
- Everything from phases 2–3 still applies: sim-only eval, relative scores, final-frame success and same-machine reproducibility.

### Backlog
- R-real-3, only with a normalization that measurably reduces |z| on short mid-motion episodes and keeps every bar.
- The uniform-junk reference check on `signal_medians` (needs real data to calibrate).
- A real Hub checkpoint test of the LeRobot adapter (needs the Hub; Sebi's Mac).
- P3-4 notes: a gated full-grid stats test, an MDE point at the observed disagreement, a published Newcombe example, φ < 0.
- Record the launch feedback (datasets where the score is wrong) as fixtures.

### In review
- (none). Phase 4 is complete (release-ready). Milestone: `docs/milestones/phase4-free-release.md`.

### Blocked
- (none). Publishing waits on Sebi (by design).

### Open questions for Sebi
1. GitHub owner → `python tools/set_github_owner.py <owner>`.
2. PyPI name `robot-report-card`: go / no-go.
3. Squash git history before the public push (recommended), or keep it.
4. The pending phase 3 Mac check.
5. Publish per `RELEASE.md`; post the launch drafts (`docs/launch/`).
6. Approve phase 5 (hosted paid tier)? The product lock ends Oct 23.

---

## Session 3 — 2026-09-25

### Shipped (QA: PHASE 3 APPROVED, `docs/reviews/phase3-code-review.md` → Re-review, `f066505`)
- `514c9fb` P3-0: phase 3 plan rev 2, research brief, spikes
- `7aae63c` P3-1 engine: split `action_tv_ratio` denominator (R-real-1); `261bc70` P3-1 bench: return_home group, low-noise sets, bar, power check (QA)
- `532cb4a` P3-2: floored `length_z` (R-real-2), N5 "(below flag threshold)" wording
- `0cdc16b` P3-3: stats library (Wilson, Clopper-Pearson, exact McNemar, Newcombe paired CI, MDE)
- `6d49fb1` P3-4: stats validation vs scipy + exact enumeration (QA; Researcher approved)
- `c429c86` P3-5: `rrc eval`, `rrc compare`; `f2d22f6` P3-6: `rrc train-bc`, `bc:` policy; `d5ae09a` P3-7: LeRobot adapter + tiny-ACT fixture
- `8ee8daf` P3-8: headline e2e (pinned mix) + `bench record` (QA; Coder approved); README "Evaluate a policy" (PM; QA ran it verbatim in a clean venv)
- `94aeac4` P3-5 fix: borderline McNemar/CI disagreement sentence (R1) + N1–N7; `f066505` QA re-review
- Reviews: `docs/reviews/phase3-code-review.md`, `docs/reviews/phase3-qa-work-review.md`

### Measured
- `pytest`: **234 passed, 3 skipped** (gated) in about 2 min 20 s (136–144 s; DoD 11 budget 150 s). `ruff` clean.
- **Headline e2e (DoD 9):** pinned mix 60 clean + 10 noise 0.25 + 10 random + 5 hesitation + 5 wrong, train seed 0,
  200 eval episodes: **A (all) 0/200 vs B (ok-and-success) 29/200, Δ +14.5 pts [+9.9, +20.0], McNemar p = 3.7e-9**.
  Over train seeds 0–4: A 0–2/200, B 15–39/200, every p ≤ 6.1e-5 (per-seed table reproduced from the built wheel in
  `docs/reviews/phase4-code-review.md` → Re-review). The gated full run reproduces brief §4 (unfiltered ≤ 5/200, filtered ≥ 100/200).
- Speed: BC training about 10 s on CPU; 200 rollouts about 7 s. README demo about 50 s.
- Linux install: `.[dev,eval]` 5.4 GB, full extras 10 GB (CUDA torch).
- **R-real-1 bars** (full builds; the mixed set is now 160 episodes = 150 + 10 return_home):

| Bar (standard noise) | Target | dev | held-out | QA-private |
|---|---|---|---|---|
| Clean vs motion junk AUROC | ≥ 0.95 | 0.998 | 1.000 | 1.000 |
| Precision / recall | ≥ 0.90 / ≥ 0.85 | 1.00 / 0.975 | 1.00 / 0.975 | 1.00 / 1.000 |
| Clean flagged (mixed) / clean-only | ≤ 5% / ≤ 5% | 0/60 / 0% | 0/60 / 1% | 0/60 / 2% |
| noise 0.1 / 0.25 / random flagged | ≥ 90% each | 10/10/10 | 10/10/10 | 10/10/10 |
| Hesitation AUROC | ≥ 0.90 | 0.993 | 1.000 | 1.000 |
| Nearmiss flagged | ≤ 20% | 0/10 | 0/10 | 0/10 |
| return_home flagged | ≤ 1/10 | 0/10 | 0/10 | 0/10 |
| Low-noise return_home flagged (binding) | ≤ 1/10 | 0/10 | 0/10 | 0/10 |
| Low-noise clean-only (reported) | – | 5% | 3% | 2% |
| Power check: old `net` formula, low set | ≥ 2/10 on some set | 1/10 | 0/10 | **2/10** |

- **P3-4 statistics validation** (all bars pass, about 9 s):
  - Wilson vs scipy: ≤ 2e-16.
  - Clopper-Pearson vs scipy: ≤ 5e-13.
  - McNemar vs `binomtest`: ≤ 2e-15.
  - Newcombe vs QA's own implementation: max diff 0 on 2,007 tables. Its exact coverage (mean / worst) is 0.946 / 0.930 at n = 50 and 0.949 / 0.946 at n = 200.
  - Simulated power at the reported MDE: 0.807–0.823 (bar ≥ 0.78).
  - Wilson coverage on a 1e-4 grid over success rates 5%–95% (mean / worst): 0.954 / 0.916 at n = 20, 0.952 / 0.931 at 50, 0.951 / 0.927 at 100, 0.950 / 0.934 at 200.
  - The Researcher's mutation test caught 11/11 planted formula bugs.

### Deviations and rulings
1. **The e2e and README demo record with `python -m robot_report_card.bench record`, not `rrc record --append`.**
   `rrc record` has no hesitation policy, and its `wrong` policy still uses the mirrored target. Accepted: `bench record` ships in the package, is tested, and is a supported command. Its flags don't change without a README update.
2. **`rreal1` seed set (16000/16500)** is a *selected* regression fixture where the old formula visibly fails (net 7/10,
   split 0/10). It is labelled that way in `spec.py`. It is not a validation set.
3. **The power check** is enforced only by the gated `RRC_BENCH_FULL=1` test. `bench check` prints it but doesn't fail on it. Accepted.
4. **The MDE sentence also prints on significant results.** DoD 2 requires it on every compare, and the e2e asserts it. It says "checkpoints" only when both sides are checkpoints, otherwise "policies".
5. **Borderline sentence (R1):** when the exact McNemar verdict and the Newcombe CI disagree near α (158 of 4,223 tables),
   `compare` says so and sets `paired.borderline`.
6. **The P3-8 e2e stays in the default suite** (headline guard, n = 200). If the budget breaks, gate the train-bc determinism test (9 s) and the ACT setup (8 s) first.
7. The standard and low-noise sets share seeds: they are two views of one sample, not independent replications.

### Known gaps
- **Determinism is proven on one machine only.** A Mac vs Linux rerun is a new sample.
- **ACT is an adapter smoke test only** (0/5 success on the tiny fixture). No ACT success claim, and Hub checkpoints are untested.
- **Success is final-frame; LeRobot's eval counts any step.** Both are reported, and the numbers differ.
- **Recipe claims need ≥ 3 training seeds.** One retrain moved success 10–15 points; `compare` covers rollout noise only.
- **`rreal1` is selected.** The fair estimate of the old formula's return-home false-flag rate on low-noise data is about
  2–3/10 on average (0–6/10 per set). Split flagged 0/60 on six fresh bases.
- Low-noise clean-only is flagged 2–5% (reported, not barred) until real noise levels are known.

### Backlog
- `rrc record` gets the bench generators (or `--mix`), and `wrong` uses an independent goal. Then point the README demo at it.
- Phase 2 leftovers: N3 (`--append` doesn't check existing `.npz` files), R-real-3 (LDLJ duration bias), and the uniform-junk
  reference check on `signal_medians`.
- P3-4 notes (Researcher):
  - a gated `RRC_STATS_FULL=1` full-grid test;
  - an MDE power point at the observed disagreement (200, 0.555, 0.22);
  - a published Newcombe worked example;
  - φ < 0 coverage.
- CPU-only torch wheel install is documented but untested (the PyTorch index is blocked in the crew container).

### In review
- (none). Phase 3 is complete. Milestone: `docs/milestones/phase3-policy-evaluation.md`.

### Blocked
- (none)

### Open questions for Sebi
1. **Mac milestone check:** pull, install `.[dev,eval,score,lerobot]`, run the README eval demo and send the numbers. Re-score the DoD 8
   dataset (ep 7 should be `ok`, ep 9 still flagged). Commands are in the milestone report.
2. **Defaults, override if you disagree:**
   - n = 200 with no early stopping;
   - final-frame success;
   - torch in the `eval` extra;
   - the ACT fixture trained in the test.
3. **Approve phase 4 (free release, community distribution)?**

---

## Session 2 — 2026-09-24

### Shipped (QA-approved; P2-3 also approved by Researcher + PM)
- Phase 1 approved by Sebi; decisions logged (Apache-2.0, lerobot optional extra, no Hub upload).
- `e4001b6` P2-0: phase 2 plan rev 3, research brief, spikes (QA approved)
- `69610f7` P2-1: phase 1 backlog (corrupt `.npz` on export → clean error; piped `rrc record` exits 141)
- `690186e` P2-2: `rrc record --append`
- `6a645dc` P2-4: pyarrow reader for LeRobot v3.0 / v2.1 / v2.0 (no lerobot, no torch)
- `c127b4c` P2-5: motion-quality signals + scoring engine; `2b0a405` P2-6: `rrc score` CLI, JSON report v1, e2e, README
- `8391c03` P2-3: benchmark v2 builder, evaluator, real-like fixtures (QA-authored)
- `56b88cd` P2-fix: QA R1–R3 (+N1/N2/N4); `9e8b8c1` P2-3 fix: independent wrong goal, randomized stall, evaluator exit codes
- Reviews: `docs/reviews/phase2-code-review.md`, `docs/reviews/phase2-bench-review.md`

### Measured
- `pytest`: **153 passed, 1 skipped** (gated) in 47.6 s. `ruff` clean.
- Final DoD 4 bars (full builds, `bench check` → `rrc score` JSON; private seeds kept outside the repo):

| Bar | Target | dev | held-out | private |
|---|---|---|---|---|
| Clean vs motion junk AUROC | ≥ 0.95 | 0.999 | 1.000 | 1.000 |
| Precision (z > 3.5) | ≥ 0.90 | 1.00 | 1.00 | 1.00 |
| Recall (z > 3.5) | ≥ 0.85 | 0.975 | 0.975 | 0.975 |
| Clean flagged, mixed | ≤ 5% | 0/60 | 0/60 | 0/60 |
| Clean-only flagged | ≤ 5% | 1% | 0% | 2% |
| noise 0.1 / 0.25 / random flagged | ≥ 90% each | 10/10/10 | 10/10/10 | 10/10/10 |
| Hesitation AUROC | ≥ 0.90 | 0.995 | 1.000 | 1.000 |
| Nearmiss flagged | ≤ 20% | 0/10 | 0/10 | 0/10 |
| **All bars** | | **PASS** | **PASS** | **PASS** |

- Report-only groups (flagged/10, AUROC vs clean; dev | held-out | private):
  - borderline noise 0.05: 0, .91 | 0, .98 | 0, .96
  - hesitation flag rate: 9 | 9 | 9
  - wobble (tracked gap): 0, .82 | 0, .82 | 0, .84
  - stall: 0, .61 | 0, .81 | 0, .61 (sim success 0/10)
  - wrong (independent goal): 0, .50 | 0, .59 | 0, .62 (sim success 0/10). Motion can't see it, which matches the honesty statement.

### Deviations from plan (QA rulings, `phase2-code-review.md`)
1. `--append` creates the session if `--out` is new: approved.
2. `session.json` stores `max_steps`; older sessions fall back to their common episode length: approved.
3. Too-short episodes are HARD: approved. After R2 the minimum is idle window + 2 frames (17 at 30 fps), and the message gives frames and seconds.
4. Frozen-joint HARD: **rejected as written, fixed (R1).** It now flags only when the reading is flat *and* the command moves (dead motor). A joint that is unused in an episode is not flagged.
5. The >50%-flagged warning is kept, but it doesn't protect against uniform junk (see gaps). The wording was added in R3.
6. A warning for fewer than 10 episodes: approved.
7. Episodes with hard flags (other than non-finite or too short) still count in the robust statistics: approved.
8. Actions are normalized by their own range when dims differ; `track_err` is skipped when dims/names differ: approved.
9. `--json` goes to stdout, the default JSON path is refused on a re-run without `--overwrite`: approved.

### Known gaps (stated in README / report where user-facing)
- **Wobble** (slow band-limited oscillation): AUROC ~0.82, flagged 0/10. Tracked gap, no bar.
- **Uniform junk:** if every episode shares the problem (e.g. a jittery leader arm), 0% are flagged. Scores are relative.
- **Smooth failed attempts** (wrong goal, early stall, nearmiss) are invisible to motion quality; only outcome evidence shows them.
- **Watch item, short reaches:** 4/200 (2%) of a wider-spread clean set were flagged "dithering commands" on
  reaches with 0.45–0.55× the median displacement (the `action_tv_ratio` floor). This is the likeliest real-data false flag; check it in DoD 8.

### Backlog
- Phase 3: a dataset-level reference check on `signal_medians` (catches uniform junk; needs real data to calibrate).
- N3: `--append` doesn't check that existing episode `.npz` files are present (export catches it later).
- N5: `ok` rows list reasons for 3 < z ≤ 3.5. Add "(below flag threshold)".
- **R-real-1 (from DoD 8, priority):** `action_tv_ratio` false-flags return-to-home tasks. Use a denominator that doesn't collapse when the arm ends near its start (e.g. max excursion from the start pose, or path split at the farthest point). Add a return-to-home fixture/benchmark group and re-check the bars.
- **R-real-2:** `length_z` explodes (-582, -709) when most episodes share one length (MAD ≈ 1, e.g. a recording time cap). It isn't scored, but the JSON value is misleading. Floor the MAD or report the length percentile instead.
- **R-real-3:** LDLJ is biased by episode duration (short real episodes get z ≈ -5 to -8). Negative only, so there are no false flags today, but normalize before any two-sided use.

### DoD 8: real-data check (Sebi, MacBook Air, 2026-09-24): DONE
- Dataset: a public SO-101 dataset (name withheld) (LeRobot v3.0, `so_follower`, 30 fps, 10 eps, 6,534 frames, 6-D state/action, one front camera; videos not downloaded).
- `rrc score` read it with no key overrides and no HARD flags. It flagged 2/10; outcome is `unknown` for all (no success column or labels, which is correct).
- Sebi watched the videos:
  - ep 7 FLAG "dithering commands" (tv_ratio z=4.1): **false positive.** Normal task where the arm returns to its start pose, so net displacement is small and path/net blows up. This is the predicted watch item, now confirmed on real data.
  - ep 9 FLAG "looks jittery" (hf_energy z=3.9): **true positive** (shaky).
  - ep 3 ok: **true negative** (looks decent).
- Not retuned in phase 2, per plan. Fixes go to the backlog below.

### In review
- (none). Phase 2 complete. Milestone: `docs/milestones/phase2-dataset-scoring.md`.

### Blocked
- (none)

### Open questions for Sebi
1. ~~**DoD 8 real-data check**~~ Done 2026-09-24: 1 true positive, 1 false positive (return-to-home), see above.
2. **Defaults applied, override if you disagree:** pyarrow is an optional `score` extra; `rrc score` never modifies
   the dataset; episode length is reported but not scored.
3. **Approve phase 3 (policy evaluation)?**

---

## Session 1 — 2026-09-24

### Shipped (QA: PHASE 1 APPROVED, see `docs/reviews/phase1-code-review.md` → Re-review)
- Repo scaffold: README, LICENSE (Apache-2.0 + NOTICE, switched from MIT per Sebi), STATUS.md, `src/` / `tests/` / `docs/` layout
- Docs: `docs/research-brief.md` (Researcher), `docs/phase1-plan.md` rev 3 (PM), README Quickstart (PM). All QA-reviewed.
- P1-1 package, `rrc` CLI, session format, vendored SO-101 assets (`2cb7ddd`, R4 fixed in `b1ec05c`)
- P1-2 `so101_reach` env, policies, `rrc record` (`f435179`, R3 fixed in `b1ec05c`)
- P1-3 `rrc tag` / `rrc list` (`272c70d`, R5 fixed in `b1ec05c`)
- P1-4 `rrc export` → LeRobot v3.0 + `meta/rrc_tags.json` (`62cb9cd`). R1 (overwrite could delete the session) and R2 (half-written output after a failure) fixed in `b1ec05c`.
- P1-5 end-to-end test `tests/test_p1_5_e2e.py` (`51f7594`, QA-owned)
- Piped-output fix: `rrc list | head` no longer crashes with BrokenPipe (`d5bdab3`)
- QA re-ran break attempts B1–B8: all now fail cleanly and no data is lost.

### Measured
- `pytest`: **72 passed in 19.3 s**. `ruff` clean.
- Clean venv with `[dev,sim]`: **about 200–260 MB** (202 MB via uv, 258 MB via pip, 22 s install). There, record/tag/list work, and export fails with an install hint.
- README Quickstart run verbatim by QA in a fresh copy: export + offline load return `10 900`.
- Demo: recording 10 episodes takes 1.1 s, export takes 5.1 s.
- `so101_reach` success (2 cm, 90 frames, `SeedSequence([seed, i])`), over seeds 0–9:
  - `scripted` (gravity compensation on): 100% (500/500).
  - `--no-gravity-comp`: 81.2%.
  - `--noise 0.05` / `0.1` / `0.25`: 70.6% / 33.6% / 8.4%.
  - `random`: 0.2%. `stall` / `wrong`: 0%.
  - Plan bars (≥60% scripted / ≤10% random) met.
- The lerobot extra on Linux: about 8.1 GB (CUDA torch).
- **macOS (Sebi, MacBook Air, Apple Silicon, conda Python 3.11):** Quickstart run verbatim and it worked. Full env with `[dev,sim,lerobot]` is **1.8 GB**. Both installs took about 10–15 s each, likely from pip's cache, so a first install will be slower. Export + offline load returned `10 900`.

### In review
- (none)

### Backlog (non-blocking, from QA re-review, for phase 2)
1. A present-but-corrupt `.npz` (wrong dtype) gives a `ValueError` traceback on `rrc export`. Cleanup is still correct. Also catch `ValueError` in `commands/export.py:49`.
2. `rrc record ... | head -1` stops with "N episodes saved" and exit 1 instead of 141. The session is still valid. Either let `BrokenPipeError` reach `cli.main`, or keep recording with output silenced.
3. README install size wording: done ("about 200–260 MB").

### Deviations from plan (QA-ruled)
- Seeding: `SeedSequence([seed, i])` instead of `seed + i`, because neighbouring seeds overlapped. Approved, and the plan is updated.
- Gravity compensation is on by default, so `scripted` succeeds 100%. Approved: `scripted` is the clean good-demo source. `--no-gravity-comp` gives near-misses, and `--noise`/`stall`/`wrong`/`random` give junk.
- A sim divergence aborts `record` instead of saving an `error` episode. Accepted on condition of R3. `termination_reason="error"` is **reserved and unused** in phase 1. None of 400 stress episodes diverged.

### Known limitations (phase 1)
- `rrc tag` has no locking: two concurrent runs on one session can lose an update.
- One policy/noise setting per session, and export works one session at a time, so there's no single dataset with mixed good and junk episodes yet (see open question 4).
- `--repo-id` isn't validated.
- State only: no cameras, video or rendering, and no Hub upload. macOS verified by Sebi (Quickstart only; full pytest suite not yet run on the Mac).

### Blocked
- (none). Phase 1 is approved. Phase 2 waits for Sebi's go-ahead (see `docs/milestones/phase1-data-logger.md`).

### Open questions for Sebi
1. ~~**Run the README Quickstart on the MacBook Air.**~~ Done 2026-09-24; see Measured. Report whether the install worked, how big it is and how long it took (both `[dev,sim]` and `[dev,sim,lerobot]`), and whether export + load work. This is the only macOS check phase 1 gets.
2. ~~**License: MIT or Apache-2.0?**~~ **Decided 2026-09-24: Apache-2.0.** LICENSE replaced, NOTICE added, `pyproject.toml` and README updated.
3. ~~**Defaults**~~ **Decided 2026-09-24: keep both.** lerobot stays an optional extra used only by export; Hugging Face Hub upload stays out of scope for phase 1.
4. **Proposed first phase-2 prerequisite:** `rrc record --append` (or a session merge), so one exported dataset can mix good and junk episodes. The scorer needs that to be validated against. Not blocking phase 1.

# Phase 4 Plan — Free Release (release-ready v0.1.0)

Author: PM · Date: 2026-09-25 · Status: rev 2, QA changes applied (see end)
Inputs: `docs/phase4-research-brief.md`, `STATUS.md`. Sebi's scope: **release-ready**. The crew builds and rehearses
everything; **Sebi presses publish** on PyPI and GitHub. The crew uploads, publishes and posts nothing.

## Goal

A 0.1.0 Sebi can publish in minutes: clean artifacts, PyPI-ready README, CI + a trusted-publishing workflow awaiting
his approval, `rrc report` (the one-page card), `rrc record --mix`, and draft launch posts.
Out of scope: publishing, creating the repo, posting, docs site, new metrics, real-robot eval, retuning.

## Definition of Done

1. **Build:** `python -m build` → sdist + wheel; `twine check dist/*` passes; metadata is PEP 639 (`License-Expression:
   Apache-2.0`, LICENSE + NOTICE + SO-101 LICENSE in `dist-info/licenses/`); version `0.1.0`.
2. **Wheel install (not editable), clean venvs:**
   base venv ≤ 100 MB [74 measured], `rrc --help`/`tag`/`list` work, `rrc score <real dataset>` without pyarrow prints
   the PyPI hint `pip install "robot-report-card[score]"`; `[score,sim]`: README Quickstart verbatim (record → tag → list → score); `[score,sim,eval,lerobot]` (existing
   venv allowed): README evaluate demo.
3. **Full test suite passes from the sdist:** tests run from the unpacked sdist against the *installed* sdist (not
   `src/`); light profile `[dev,score,sim]` in a clean venv, full profile in a venv with the heavy extras + scipy.
4. **CI files exist and are lint-valid:** `.github/workflows/tests.yml` (light matrix + heavy + weekly
   `RRC_BENCH_FULL=1`) and `release.yml`. `actionlint` passes if installable (e.g. `actionlint-py`); otherwise YAML
   parses and QA reviews by hand. README/CONTRIBUTING say plainly: **"CI has not yet run on GitHub."**
5. **Release workflow:** trusted publishing (OIDC, `id-token: write`, `pypa/gh-action-pypi-publish`), triggered on
   `release: published`; the publish job is bound to `environment: pypi`; a `workflow_dispatch` TestPyPI job for the
   dry run; no tokens or secrets. RELEASE.md tells Sebi to (a) create the `pypi` environment with himself as required
   reviewer and (b) register the pending publisher with exactly `<owner>/robot-report-card`, `release.yml`, `pypi`.
   QA checks the YAML, not the GitHub settings.
6. **`rrc record --mix`** (e.g. `clean:60,noise025:10,random:10,hesitation:5,wrong:5 --seed 30000`) produces a session
   **byte-identical** to `python -m robot_report_card.bench record` with the same args (arrays + `episodes.jsonl` apart
   from `recorded_at`). `--list-groups` prints each group with its class. `--policy wrong` now uses an **independent
   goal**; `--wrong-goal mirrored` restores phase 1 behavior; `policy_params.goal` records which.
7. **`rrc report`** builds the one-page report card from `rrc score` / `rrc eval` / `rrc compare` JSON: blocks Data,
   Policy, Regression, Verdict, **Can't tell**; terminal always, `--md FILE`, `--html FILE` (stdlib template, inline
   CSS, no JS), `report.json` as source of truth. Rules, dataset linking, fixed Can't-tell lines and forbidden phrases
   exactly as in "Report-card rules (D2)" below. **QA's rules-table test covers every rule (R0–R6 + inconclusive),
   linked/unlinked, and every forbidden phrase**, in all three renderings.
8. **Docs:** README rewritten for PyPI (hero, absolute links, install lines incl. CPU-torch guidance, phase results
   with honest caveats, "What it can't tell you"); `CHANGELOG.md` (0.1.0), `CONTRIBUTING.md`, `SECURITY.md` (loading
   checkpoints), `RELEASE.md` (Sebi's publish checklist).
9. **Launch drafts** in `docs/launch/`: `discord.md`, `hf-forum.md`, `x-thread.md`, per D5. Every number traces to
   STATUS; no dataset, org or person is named except our own; drafts say DRAFT at the top. Sebi posts.
10. **`GITHUB_OWNER`** is the only placeholder, `tools/set_github_owner.py <owner>` replaces every occurrence, and
    `tools/release_check.py` fails while it is still present.
11. Default `pytest` stays < 2.5 min (144 s today); `ruff` clean. If it goes over: gate the P3-6 cross-process
    determinism test and the tiny-ACT eval tests behind `RRC_SLOW=1`, never the P3-8 headline.
12. **Third-party data (D6):** no tracked file names the DoD 8 dataset; `release_check.py` fails if any entry of a
    redaction denylist (kept outside the repo, passed by path or env var) matches a tracked file.

## Design decisions

- **Packaging (brief §1):** `hatchling>=1.27`, `license = "Apache-2.0"`, `license-files`, keywords, classifiers
  (no `License ::` classifier), `[project.urls]` with `GITHUB_OWNER`. Move `numpy<2.4` into the `lerobot` extra (base
  keeps `numpy>=1.25`). Extras unchanged. Classifiers 3.10–3.12 (light suite verified by QA on all three; README says
  the heavy extras are tested on 3.11). **Every user-facing install hint** reads `pip install "robot-report-card[<extra>]"`
  (5 sites: `sim/registry.py`, `score/reader.py`, `eval/bc.py`, `eval/lerobot_adapter.py`, `export/lerobot_writer.py`),
  and a test greps `src/` for `pip install -e`.
- **sdist tests:** ship `docs/spikes/phase2_score_proto.py` and `phase3_stats.py` in the sdist (tests check parity
  against them) rather than skipping those tests.
- **One placeholder:** the literal `GITHUB_OWNER` appears in `pyproject.toml` URLs, README links, CHANGELOG/CONTRIBUTING
  links and launch drafts. A test asserts it appears only in an allowlist of files, so `set_github_owner.py` is complete.
- **`tools/release_check.py`** (also used by QA and `RELEASE.md`): build → twine check → base/`[score,sim]` wheel venvs +
  Quickstart smoke → sdist light suite → placeholder check. Prints a PASS/FAIL table. It never uploads.
- **`rrc report`:** `--train-seeds N` (default 1) drives the recipe caveat. It prints which files it used and warns if
  eval/compare seeds or policies don't line up.
- **`--mix` (D3):** requires `--max-steps 90` (generators assume 3 s episodes); `--list-groups` shows `return_home` as
  `clean_variant`; bench generators and group names are a **public API from 0.1.0** (docstring + CHANGELOG).
  `--policy wrong` defaults to the independent goal (help text and README updated).
- **`--mix`** calls the same `bench.generators` + table parser; `bench record` becomes a thin alias. Mutually exclusive
  with `--policy/--episodes/--noise`; `--noise-profile standard|low`; works with `--append`.

## Report-card rules (D2, binding for P4-4 and P4-5)

- **Data block (always):** HARD count by flag type; the HARD line names the types present ("N episodes have hard
  flags: non-finite values / dropped frames / too short / frozen joint / saturated actions"), never "corrupt".
- **Linking:** a `bc:` checkpoint is *linked* when its `rrc_policy.json` dataset fingerprint (path + total_frames)
  matches the score JSON's dataset; `lerobot:` and scripted policies are unlinked. Only linked verdicts may call scored
  episodes "training demos".
- **Rules, first match wins** (thresholds are named constants; footer says "sim-tuned proposals"):
  - **R0** HARD ≥ 5% of episodes, or any non-finite → "Fix the data first: N episodes have hard flags (types…)."
  - **R1** flagged ≥ 50% → "Most episodes are flagged. The score is relative and assumes most episodes are good, so
    per-episode flags aren't reliable here."
  - **R2** outcome evidence and failed ≥ 10% → "X% of demos in <dataset> failed (outcome source …). Consistent with a data
    problem [linked: in the data this policy was trained on]. In our sim study, training on failed wrong-goal demos
    cost ~24 points."
  - **R3** motion-flagged ≥ 10% → "X% of demos flagged (reasons with counts). Consistent with a data problem. In our sim
    study, random and high-noise demos hurt the policy; hesitation didn't."
  - **R4** compare not significant, or borderline → "Eval can't tell A from B (Δ, CI, p). With n = N it detects
    differences ≥ MDE." (+ the borderline sentence when flagged)
  - **R5** compare significant and not borderline, both checkpoints linked, flagged < 5%, outcome known with failed
    < 5% → "A and B differ in this eval (Δ, CI, p). The scorer found no problems in the data both were trained on; it
    can't see uniform junk, wobble or smooth failed attempts. So the difference is consistent with training variance or
    the recipe." (+ the ≥ 3 training-seeds caveat if `--train-seeds` < 3). Unlinked → R5 doesn't apply.
  - **R6 (add-on)** no outcome evidence → "Outcome unknown: wrong-goal and early-stop demos can't be ruled out."
  - **else** → "Inconclusive: missing / unlinked …"
- **Can't tell (always, fixed lines):** uniform junk gives few or no flags; wobble/band-limited jitter isn't flagged;
  smooth failed attempts need outcome evidence; checkpoint vs recipe (10–15 pts training-seed variance, ≥ 3 seeds);
  success is final-frame and LeRobot's any-step rate is shown alongside and differs; eval is in simulation
  (`so101_reach`), not on a real robot; scores aren't comparable across datasets; reproducibility is same-machine only;
  thresholds are sim-tuned.
- **Forbidden phrases** (all renderings): "caused by", "because of", "data looks clean", "data is good", "clean data",
  "proves", "real robot will", "reproducible across", any recipe verdict with < 3 seeds, any ACT/LeRobot quality adjective.

## Subtasks (ordered)

### P4-1 Packaging metadata + release check
- **Scope:** pyproject per design (D1); version 0.1.0; sdist includes the two spikes; PyPI install hints + grep test;
  `GITHUB_OWNER` placeholder + `tools/set_github_owner.py` + allowlist test; `tools/release_check.py` incl. the
  redaction-denylist check.
- **Acceptance:** DoD 1, 2 (base + `[score,sim]` parts), 3, 10, 12 on this machine; `release_check.py` passes except the
  placeholder check (expected to fail until Sebi sets the owner, and says so).
- **Depends on:** nothing. **Does:** Coder. **Reviews:** Researcher (packaging), QA.

### P4-2 CI + trusted-publishing workflows
- **Scope:** `tests.yml` (light: ubuntu + macos-14 × 3.10–3.12, `.[dev,score,sim]`, ruff + pytest; heavy: 3.11 ubuntu,
  CPU torch pinned `torch==2.10.*` from the CPU index first, full extras + scipy, on main pushes; weekly
  `RRC_BENCH_FULL=1`), `release.yml` (PyPI via `pypi` environment + TestPyPI via `workflow_dispatch`), `HF_HUB_OFFLINE=1`.
- **Acceptance:** DoD 4, 5; actions pinned to major versions; no secrets referenced; permissions least-privilege
  (`contents: read`, `id-token: write` only on the publish job).
- **Depends on:** P4-1. **Does:** Coder. **Reviews:** Researcher, QA.

### P4-3 `rrc record --mix` + independent `wrong`
- **Scope:** DoD 6 + D3; `--list-groups`; `bench record` alias; tests pinning the old `wrong` get `--wrong-goal mirrored`.
- **Acceptance:** byte-identical test vs `bench record` on a **small mix** (`clean:3,noise025:1,random:1,hesitation:1,
  wrong:1,return_home:1`), standard and low profile; `--mix` + `--policy` and `--mix` with `--max-steps` ≠ 90 refused
  cleanly; unknown group / zero count refused; old sessions still load; no bar re-run needed (bench already independent).
- **Depends on:** nothing (parallel with P4-1). **Does:** Coder. **Reviews:** QA.

### P4-4 `rrc report`
- **Scope:** `robot_report_card/report/` (rules engine per D2, linking, `report.json`, terminal/Markdown/HTML) and the CLI.
- **Acceptance:** P4-5 passes; the phase 3 demo JSONs produce a sensible card (QA reads it); an HTML file opens with
  no network; missing or mismatched inputs give clean errors or the "Inconclusive: missing …" verdict.
- **Depends on:** nothing. **Does:** Coder. **Reviews:** QA (via P4-5), Researcher (rule wording vs evidence).

### P4-5 Report-card rules test (QA-owned)
- **Scope:** `tests/test_p4_5_report_rules.py`, table-driven from synthetic JSON inputs: one row per D2 rule (R0–R6,
  inconclusive), precedence, R6's add-on, linked vs unlinked (R5 only when linked; "training demos" only when linked),
  HARD wording by type. The forbidden-phrase list runs on all three renderings; every fixed Can't-tell line is present.
- **Acceptance:** written from brief §3 before or alongside P4-4 (test-first allowed); every rule row fails if its rule
  is removed (QA mutation check); < 10 s.
- **Depends on:** the spec only. **Does:** QA. **Reviews:** Researcher.

### P4-6 Backlog (cheap only)
- **Scope:** N3 (`--append` checks existing `.npz` files); R-real-3 (LDLJ duration normalization) **only if** the bench
  bars on dev / held-out / private stay unchanged; otherwise defer and say so.
- **Acceptance:** tests for N3; for R-real-3 either a before/after bar table or a documented deferral.
- **Depends on:** nothing. **Does:** Coder. **Reviews:** QA.

### P4-7 Docs for release
- **Scope:** README rewrite (DoD 8; demo moves to `rrc record --mix`; a short `rrc report` example), CHANGELOG,
  CONTRIBUTING (dev install, test profiles, gated tests, "CI not yet run on GitHub"), SECURITY (checkpoints: `bc:` uses
  `weights_only=True` + SHA-256 check; `lerobot:` loads through lerobot, so only load checkpoints you trust; how to
  report issues), RELEASE.md (set owner → release_check → TestPyPI dry run → create repo → register pending publisher
  → create release → approve `pypi` environment).
- **Acceptance:** QA runs README install/Quickstart/demo/report blocks verbatim from the wheel; all links absolute;
  every number matches STATUS; STATUS gains the 5-seed P3-8 numbers; RELEASE.md includes the environment/publisher
  steps (DoD 5) and the history decision (Needs Sebi's call 4).
- **Depends on:** P4-1, P4-3, P4-4. **Does:** PM. **Reviews:** QA, Researcher (install/packaging text).

### P4-8 Launch drafts
- **Scope:** DoD 9. Lead with the LeRobot Discord, then the HF forum, then an X thread (≤ 6 posts). Hook: the
  community-dataset quality problem. Facts (D5): "on a public SO-101 dataset (not named), 1 true positive and 1 false
  positive (return-to-home); the false positive is fixed and verified in simulation, the re-check on the real dataset
  is pending" (unless Sebi's re-score is in STATUS first); "two checkpoints trained on the same data, unfiltered vs
  filtered: 0/200 vs 29/200 (p = 3.7e-9)", with the data claim citing the 5-seed result (A 0–2/200, B 15–39/200,
  p ≤ 6e-5); always "in simulation, one task". Honest limits; the ask ("send us a dataset or `rrc score` JSON where it's wrong").
- **Acceptance:** QA checks every claim against STATUS and the forbidden-claims list; no third-party names.
- **Depends on:** P4-7. **Does:** PM. **Reviews:** QA.

### P4-9 Release rehearsal
- **Scope/acceptance:** QA clones fresh, runs `set_github_owner.py example-owner` in a scratch copy, then `release_check.py`,
  and walks RELEASE.md up to (not including) any upload; every DoD item recorded in STATUS with the suite time.
- **Depends on:** P4-1..P4-8. **Does:** QA. **Reviews:** PM (milestone).

## Risks

| Risk | Mitigation |
|---|---|
| CI and release workflows never ran on GitHub (no access here) | actionlint/YAML checks; state it in README and RELEASE.md; Sebi's first push is the real test |
| Name taken or too similar on PyPI; TestPyPI blocked here | TestPyPI dry run is step 1 of RELEASE.md; name is a decision below |
| Heavy CI job pulls CUDA torch (lerobot's pin not met by CPU wheel) | Pin `torch==2.10.*` from the CPU index first; unverified here (index blocked) |
| `rrc report` over-claims attribution | D2 rules (linking, typed HARD, no "clean data"), forbidden-phrase test (QA), fixed Can't-tell lines |
| Third-party dataset name in git history (263fa31 onwards) | Redacted in tracked files; denylist check; history decision is Sebi's (below) |
| Suite budget (144 s of 150 s) | Small-mix `--mix` test; P4-5 < 10 s; `RRC_SLOW=1` gating order per DoD 11 |
| Changing `--policy wrong` breaks existing tests or bars | Only tests pinning the old behavior change; bench already uses independent goal; bars re-run |
| Launch post names someone's flagged data | Rule in DoD 9; QA review; public examples use our sim data or Sebi's own |
| macOS only tested by Sebi by hand | CI macos-14 light job; Mac check below |
| Scope (9 subtasks) | Each is small; P4-6 optional; `--html` slips first |

## Defaults (Sebi can override at milestone)

- Version 0.1.0, Alpha, Apache-2.0; trusted publishing with a manual-approval `pypi` environment (`twine` fallback).
- `rrc report` HTML via a stdlib template, Markdown recommended; `--policy wrong` = independent goal from 0.1.0.

## Needs Sebi's call

1. **GitHub owner and repo name** (fills `GITHUB_OWNER`; confirms the repo is `robot-report-card`).
2. **PyPI name `robot-report-card`:** looks free (404); only a TestPyPI/PyPI upload settles it. Proceed with it?
3. **Before publishing, on your Mac:** the pending phase 3 check plus `release_check.py` (Mac wheel install), per RELEASE.md.
4. **Git history for the public repo:** start from a squashed snapshot (clean history), or keep history (the DoD 8
   dataset name stays visible in commits from 263fa31 onwards)?

## QA Review (phase 4)

QA, 2026-09-25. **Verdict: CHANGES REQUIRED (spec-level; building can start).** The packaging plan is sound: QA re-ran it (see the brief's QA review).
- base wheel venv: 74 MB;
- `[score,sim]`: 322 MB;
- `twine check` passes, with PEP 639 metadata;
- the light suite from the sdist passes on Python 3.10, 3.11 and 3.12.

What must change before building is the report-card rules, one DoD item that can't be tested here, the suite budget, and the handling of third-party data.

**Findings:**
1. **DoD 5: "requires Sebi's manual approval" can't be put in YAML.** Environment protection rules (required reviewers) are a GitHub repo setting, and trusted publishing has to match the workflow file name and environment name exactly. Both belong in RELEASE.md as Sebi's steps, and QA can only check that `release.yml` names `environment: pypi`.
2. **DoD 2:** "`rrc score` without pyarrow prints the install hint" only holds when it's given a real dataset path. On a nonexistent path the dataset check fires first. Test it with a dataset. The hint text itself is wrong for PyPI users (brief review, finding 1).
3. **DoD 7 / brief §3: the rules over-claim** (details in the brief review, finding 2). Replace them with D2.
4. **DoD 11: the budget is nearly spent** (144 s of 150 s). A byte-identical `--mix` test on the full demo mix records 2 × 90 episodes (about 10–12 s) and would break it on its own.
5. **P4-3:**
   - `--mix` makes the `bench` generators part of the public CLI, so their group names and definitions become an API.
   - The generators assume 90-frame episodes (ramps ≤ 2.5 s; return_home ≤ 2.8 s).
   - The `wrong` switch is low risk: the bench bars and P3-6/P3-8 are unaffected, and the one test using `make_policy("wrong")` asserts failure, which still holds.
6. **DoD 9 / P4-8:** "DoD 8 FP since fixed" is verified in sim only. "0/200 → 29/200" is one checkpoint pair, not a recipe result (brief review, finding 4).
7. **Third-party data:** `STATUS.md` names the DoD 8 Hub dataset (a public SO-101 dataset, name withheld) and calls its ep 9 "shaky". It's in git history from 263fa31. A public repo would publish that.
8. **Plan vs. brief:** consistent otherwise. The plan's `GITHUB_OWNER` placeholder, mismatch warnings and `tools/release_check.py` improve on the brief. The heavy CI torch pin and the macOS job stay unverified here (proxy), as both documents say.
9. **Scope: fits one session.** P4-4 is the only sizeable new code. P4-2 (CI) and P4-7/P4-8 (docs) are text. `--html` slips first, as the plan says. The rewritten rules (D2) are simpler to test than the originals.

**Required changes:**
- R1 (DoD 5): change to "`release.yml` binds the publish job to `environment: pypi`; RELEASE.md tells Sebi to (a) create that environment with himself as required reviewer, and (b) register the pending publisher with exactly `<owner>/robot-report-card`, `release.yml` and `pypi`." QA checks the YAML, not the setting.
- R2 (DoD 2): run the no-pyarrow check on a real dataset path. The expected text is the PyPI form (D1).
- R3 (DoD 7): implement D2. P4-5 tests D2.
- R4 (DoD 11): per D4.
- R5 (P4-3): per D3.
- R6 (DoD 9, P4-8, RELEASE.md): per D5 and D6.

**Reconciled decisions (binding):**
- **D1 Packaging (P4-1):**
  - pyproject as brief §1 (verified), version 0.1.0;
  - `numpy<2.4` moves to the `lerobot` extra;
  - classifiers for 3.10, 3.11 and 3.12 (light profile verified; README says the heavy extras are tested on 3.11);
  - sdist ships `docs/spikes/phase2_score_proto.py` and `phase3_stats.py`.
  - **Every user-facing install hint** says `pip install "robot-report-card[<extra>]"`, and a test greps `src/` for `pip install -e` (the files in finding 1 of the brief review).
- **D2 Report-card rules (P4-4 builds, P4-5 tests):**
  - **Data block, always shown:** the HARD count broken down by flag type. The HARD line describes the flag types present ("N episodes have hard flags: non-finite values / dropped frames / too short / frozen joint / saturated actions"), never "corrupt" in general.
  - **Dataset link:** the report checks whether each `bc:` checkpoint's `rrc_policy.json` dataset fingerprint (path and total_frames) matches the score JSON's dataset. "Linked" = matches; LeRobot and scripted policies are unlinked. Verdicts may call the scored episodes "training demos" only when linked.
  - **Rules, first match wins:**

    | # | Condition | Verdict |
    |---|---|---|
    | R0 | HARD ≥ 5% of episodes, or any non-finite | "Fix the data first: N episodes have hard flags (types…)." |
    | R1 | flagged ≥ 50% | "Most episodes are flagged. The score is relative and assumes most episodes are good, so per-episode flags aren't reliable here." |
    | R2 | outcome evidence and failed ≥ 10% | "X% of demos in <dataset> failed (outcome source …). Consistent with a data problem [if linked: in the data this policy was trained on]. In our sim study, training on failed wrong-goal demos cost ~24 points." |
    | R3 | motion-flagged ≥ 10% | "X% of demos flagged (reasons with counts). Consistent with a data problem. In our sim study, random and high-noise demos hurt the policy; hesitation didn't." |
    | R4 | compare not significant, or borderline | "Eval can't tell A from B (Δ, CI, p). With n = N it detects differences ≥ MDE." Add the borderline sentence when flagged. |
    | R5 | compare significant (not borderline), both checkpoints linked to the scored dataset, flagged < 5%, outcome known with failed < 5% | "A and B differ in this eval (Δ, CI, p). The scorer found no problems in the data both were trained on; it can't see uniform junk, wobble or smooth failed attempts. So the difference is consistent with training variance or the recipe." Plus the ≥ 3 training seeds caveat if `--train-seeds` < 3. If not linked, R5 doesn't apply and the report is inconclusive about the cause. |
    | R6 (add-on) | no outcome evidence | "Outcome unknown: wrong-goal and early-stop demos can't be ruled out." |
    | else | | "Inconclusive: missing / unlinked …" |

  - Thresholds are named constants, and the footer says "sim-tuned proposals".
  - **"Can't tell" block, always present, with fixed lines:**
    - uniform junk (every episode shares a problem → few or no flags);
    - wobble / band-limited jitter isn't flagged;
    - smooth failed attempts need outcome evidence;
    - checkpoint vs. recipe (10–15 pts training-seed variance, ≥ 3 seeds);
    - **success is final-frame; LeRobot's any-step rate is shown alongside and differs**;
    - eval is in simulation (`so101_reach`), not on a real robot;
    - scores aren't comparable across datasets;
    - reproducibility is same-machine only;
    - thresholds are sim-tuned.
  - **Forbidden phrases** (P4-5 checks all three renderings): "caused by", "because of", "data looks clean", "data is good", "clean data", "proves", "real robot will", "reproducible across", any recipe verdict without ≥ 3 seeds, and any ACT/LeRobot quality adjective.
- **D3 `--mix` (P4-3):**
  - as the plan says, plus: `--mix` requires `--max-steps 90` and refuses other values (the generators assume 3 s episodes);
  - `--list-groups` shows `return_home` as `clean_variant`;
  - the `bench` generators and group names are a public API from 0.1.0 (docstring and CHANGELOG);
  - `--policy wrong` defaults to the independent goal, with `--wrong-goal mirrored`, `policy_params.goal` recorded, and the help text and README updated;
  - bench and P3 bars are unaffected, so there's no re-run requirement beyond the existing suite.
- **D4 Suite budget:**
  - the byte-identical test uses a small mix (e.g. `clean:3,noise025:1,random:1,hesitation:1,wrong:1,return_home:1`) on the standard and low profiles;
  - P4-5 stays < 10 s;
  - if the default suite goes over 150 s, gate `test_same_seed_same_weights_and_rollouts_across_processes_and_threads` (P3-6) and the tiny-ACT eval tests behind `RRC_SLOW=1`, **not** the P3-8 headline.
- **D5 Launch drafts (P4-8):**
  - DoD 8: "on a public SO-101 dataset (not named), 1 true positive and 1 false positive (return-to-home). The false positive is fixed and verified in simulation; the re-check on the real dataset is pending", unless Sebi's re-score has landed in STATUS first.
  - The phase 3 result is "two checkpoints trained on the same data, unfiltered vs. filtered: 0/200 vs. 29/200 (p = 3.7e-9)". The data claim cites the multi-seed result (5 training seeds: A 0–2/200, B 15–39/200, p ≤ 6e-5; add those to STATUS in P4-7). Always "in simulation, one task".
  - No dataset, org or person is named except our own.
- **D6 Third-party data (P4-7, RELEASE.md, Needs Sebi's call):**
  - redact the dataset name in STATUS.md ("a public SO-101 Hub dataset, name withheld") and anywhere else it appears;
  - `release_check.py` fails if a redaction denylist entry (kept out of the repo, passed by path or env var) matches any tracked file;
  - **Sebi decides** whether the public repo starts from a squashed snapshot (clean history) or keeps history (the dataset name stays visible in 263fa31 onwards). Add this as Needs Sebi's call 4.

### Changes applied (PM, 2026-09-25, rev 2)
- R1: DoD 5 binds the publish job to `environment: pypi`; required reviewer and pending-publisher setup move to RELEASE.md as Sebi's steps.
- R2/D1: DoD 2 checks the no-pyarrow hint on a real dataset with the PyPI form; the design lists the 5 hint sites and a `pip install -e` grep test; classifiers are 3.10–3.12.
- R3/D2: new "Report-card rules (D2)" section: linking, typed HARD, R0–R6 rewritten, fixed Can't-tell lines, forbidden phrases. DoD 7, P4-4 and P4-5 point to it.
- R4/D4: DoD 11 now has the gating order; P4-3 uses a small-mix byte-identical test.
- R5/D3: `--mix` requires `--max-steps 90`, `return_home` is shown as `clean_variant`, and the generators are a public API.
- R6/D5/D6: P4-8 facts corrected (sim-only fix, checkpoint pair + 5-seed result). New DoD 12 covers redaction and the denylist check. Needs Sebi's call 4 is the history decision. STATUS.md and this review are redacted.

# Phase 4 Code Review

QA, 2026-09-25. (Initial status note below; the review of the landed code follows under **Review of the landed code**.) Status at first writing: **waiting for the Coder's P4-1..P4-4 and P4-6.** At the time of writing, no Coder commit has landed after `20e6606` (P4-0). QA waited about 30 minutes and found no new commits, branches, worktrees or uncommitted changes.

## Done by QA

- **P4-5 (`8dcb33d`):** `tests/test_p4_5_report_rules.py`, test-first. It skips until `robot_report_card.report` exists.
  - It has 21 rule rows, from D2.
  - Inputs are made by the real phase 2/3 JSON builders, so they're schema-exact.
  - The synthetic compares were sanity-checked:
    - "significant": p = 4e-10, CI (+0.13, +0.25);
    - "not significant": p = 0.83;
    - "borderline": p = 0.0625 with the CI (+0.002, +0.050) excluding 0.
- **Redaction (D6), checked against `/home/claude/.rrc_redactions.txt` (2 entries):**
  - **0 tracked files** match;
  - the entries still appear in **2 commits** of history (added, then redacted in `20e6606`);
  - the history decision is Sebi's (Needs Sebi's call 4).

## The interface P4-5 assumes (the plan leaves it open; the Coder may differ, and QA will adapt `run_report` / `rule_ids`)

```
rrc report --score SCORE.json [--eval EVAL.json ...] [--compare COMPARE.json] [--train-seeds N]
           [--md CARD.md] [--html CARD.html] [--json-out REPORT.json]
```
- The terminal card always goes to stdout, and the exit code is 0 for any verdict (including "Inconclusive").
- `report.json["verdict"]` is an **ordered list of lines** `{"rule": "R0".."R6" | "INCONCLUSIVE", "text": str}`. The first line is the winning rule; R6 follows as an add-on.
- Each verdict line's `text` appears **verbatim** in the terminal, Markdown and HTML output (report.json is the source of truth).
- Linking reads `compare.A/B.policy.dataset.{path,total_frames}` (bc checkpoints) against `score.dataset.{path,total_frames}`. LeRobot and scripted policies are unlinked.
- "Flagged" for R1 counts FLAG + HARD (the score summary's `flagged`). "Motion-flagged" for R3 counts FLAG only. R5's "flagged < 5%" uses the summary's `flagged`.

A difference in flag names or the JSON layout isn't a defect, because the plan doesn't fix them. A difference in **rules, linking, fixed Can't-tell lines or forbidden phrases** is a required change: the plan's D2 wins.

## Pending (reviewed when they land)

| Subtask | Author | Status |
|---|---|---|
| P4-1 packaging + release_check | Coder | not landed |
| P4-2 CI + release workflows | Coder | not landed |
| P4-3 `rrc record --mix`, independent `wrong` | Coder | not landed |
| P4-4 `rrc report` | Coder | not landed (P4-5 will run against it) |
| P4-6 backlog (N3, R-real-3) | Coder | not landed |
| P4-9 release rehearsal | QA | blocked on P4-1..P4-4 |

## Review of the landed code

QA, 2026-09-25. Scope:
- 6dd71de (P4-1);
- 7ceaaf5 (P4-2);
- eb0254a (P4-3);
- 2e01a7e (P4-4);
- 6d2f350 (P4-6).

All are checked against `docs/phase4-plan.md` rev 2, including "Report-card rules (D2)". The Coder's numbered "deviations 5–7" list wasn't found in the repo or the commit messages, so QA identified the P4-4 deviations from the code (listed below).

**Suite:** 294 passed, 3 skipped in 2 min 18 s. `ruff` clean. P4-5: 23/23.

### Verdicts

| Subtask | Verdict |
|---|---|
| P4-1 packaging, hints, `set_github_owner.py`, `release_check.py` | **CHANGES REQUIRED** (R3; small) |
| P4-2 CI + release workflows | **CHANGES REQUIRED** (R2; small) |
| P4-3 `rrc record --mix`, `--list-groups`, independent `wrong` | **APPROVED** |
| P4-4 `rrc report` | **CHANGES REQUIRED** (R1) |
| P4-5 rules test (QA) | passes 23/23; the mutation check catches every rule (below); needs one new row after R1 |
| P4-6 N3 + R-real-3 deferral | **APPROVED**; the deferral is upheld (see R-real-3) |
| P4-9 release rehearsal (QA) | **PASS** (below) |

### P4-1

- The pyproject matches D1. The hints go through `hints.install_hint`, and a test greps `src/` for `pip install -e`.
- `set_github_owner.py` is safe:
  - the owner name is validated against GitHub's rules;
  - it refuses a placeholder outside the allowlist;
  - it edits files in place and never touches git.
- `release_check.py` **never uploads**: it only runs build, `twine check`, clean venvs and pytest; there's no `upload` or `publish` call anywhere in `tools/`. Its denylist output gives entry numbers and file:line, and never the matching text, except in two places (R3).

### P4-2

- `release.yml`:
  - OIDC trusted publishing;
  - `id-token: write` only on the two publish jobs;
  - `contents: read` elsewhere;
  - `environment: pypi` / `testpypi`;
  - no secrets;
  - a header comment gives Sebi's GitHub steps.
- `tests.yml` pins CPU torch before the extras. Both files say they haven't run on GitHub yet.
- The gap is R2.

### P4-3

Verified from the wheel:
- `--list-groups` is readable, and `return_home` is `clean_variant`.
- `rrc record --mix clean:60,noise025:10,random:10,hesitation:5,wrong:5 --seed 30000` gives **the same weight hashes** (`48b770…`, `24d70c…`) as the phase 3 `bench record` run. That's end-to-end evidence of the byte-identical session, on top of the Coder's test.
- `--policy wrong` uses an independent goal (`goal`, `goal_qpos` recorded), and `--wrong-goal mirrored` is kept.

### P4-4

- The rules match D2:
  - typed HARD wording, never "corrupt";
  - R1 counts FLAG + HARD;
  - R4 includes borderline;
  - R5 requires both sides linked, outcome evidence for every episode, and flagged / failed < 5%, and lists the unmet conditions under Inconclusive;
  - R6 is an add-on;
  - the nine fixed "Can't tell" lines are always present;
  - the thresholds footer is shown.
- HTML escapes user text: an injected `<script>` in a policy spec comes out as `&lt;script&gt;`. There's no JS and no links.
- **Mutation check (P4-5 acceptance)**, disabling each rule in `rules.py` in turn:

  | Mutation | P4-5 failures |
  |---|---|
  | R0 | 3 |
  | R1 | 1 |
  | R2 | 3 |
  | R3 | 2 |
  | R4 | 2 |
  | R5 | 3 |
  | R6 | 2 |
  | linking (always linked) | 3 |
  | a Can't-tell line removed | 18 |

  Every mutation is caught. `rules.py` was restored from git afterwards.
- **Deviations found and ruled:**
  - (a) `Inconclusive` lists R5's unmet conditions: **approved** (more informative).
  - (b) `--json-out` defaults to `./rrc_report.json` and there's an `--overwrite` flag: **approved**.
  - (c) Inline dark-mode CSS in the HTML: **approved**.
  - (d) Linking is computed only for the compare's A/B sides, so an evals-only report never links a `bc:` checkpoint: **approved** as conservative.
  - (e) **Linking ignores the checkpoint's training filter: rejected (R1).**

### Required changes

1. **R1 (P4-4, blocker): linking must respect the training filter.**
   - `link()` compares only the dataset path and `total_frames`. A checkpoint trained with `--keep ok-and-success` is therefore "linked", and the verdict then claims things about data it never saw.
   - QA reproduced this on the real phase 3 demo JSONs, from the wheel:
     - `[R2] 25.6% of demos in mix failed … Consistent with a data problem in the data this policy was trained on.`
     - But B (`bc:ckpt/filtered`, 59/90 episodes) was trained **only on quality-ok, successful episodes**. The failed demos the verdict cites were excluded from its training data.
     - R5's "the data both were trained on" has the same flaw.
   - Fix, either of:
     - (a) put the checkpoint's kept episode indices into its eval/compare policy metadata, and compute R2/R3/R5 over each linked checkpoint's kept episodes; or
     - (b) treat a checkpoint whose `filter.keep` ≠ `all` as *not* linked for the "trained on" wording and for R5, and say so ("B was trained on a filtered subset (keep = ok-and-success, 59 of 90 episodes)").
   - Also word R2 for several linked checkpoints ("the data these checkpoints were trained on").
   - QA will add P4-5 rows: a filtered checkpoint must never get "trained on" and must never satisfy R5.
2. **R2 (P4-2): make the release workflow fail on the placeholder.** PyPI never allows re-uploading a version. If Sebi forgets `set_github_owner.py`, 0.1.0 is published for good with `GITHUB_OWNER` links. The `build` job must fail if the placeholder is present. `python tools/set_github_owner.py x --check` currently exits 0, so use a grep or the placeholder step of `release_check.py`. Add a static test for it.
3. **R3 (P4-1): stop two denylist leaks in `release_check.py`.**
   - (a) A match in a tracked *file name* prints `entry #k: file name <path>`, and the path contains the entry. Print `entry #k: a tracked file name (index N)` instead.
   - (b) `DEFAULT_DENYLISTS` hard-codes the crew's `/home/claude/.rrc_redactions.txt`, a machine-specific path, in a public tool. Keep `--redactions`, `$RRC_REDACTIONS` and `~/.rrc_redactions.txt` only.

**Non-blocking:**
- N1: `release_check.py` could also print the *number* of history commits containing a denylist entry (not the entries), as the D6 reminder. Today: 2 commits.
- N2: `OWNER_RE` accepts consecutive hyphens, which GitHub rejects. Trivial.

### P4-9 release rehearsal (QA): PASS

- Setup: a fresh `git clone` of `6d2f350` into a temp dir, then `tools/set_github_owner.py example-owner` in the clone only. It replaced 3 × `GITHUB_OWNER` in `pyproject.toml`; the PM's docs aren't in yet, and `--check` then reported none left.
- `tools/release_check.py`, 37 s:

  | Step | Result | Detail |
  |---|---|---|
  | build | PASS | 6.1 MB wheel |
  | twine | PASS | |
  | metadata | PASS | Metadata-Version 2.5, Apache-2.0, 3 license files |
  | sdist | PASS | 107 files incl. tests and the spikes |
  | base-venv | PASS | 74 MB; `--help`, tag/list, score hint |
  | quickstart | PASS | 325 MB; record → tag → list → score |
  | sdist-suite | PASS | 234 passed / 27 skipped |
  | placeholder | PASS | |
  | redactions | PASS | 2 entries, 0 matches in 134 tracked files |

  **RELEASE CHECK PASSED.** Nothing was uploaded.
- **Eval demo from the wheel** (clean uv venv, `[score,sim,eval,lerobot]`): `rrc record --mix` → export → score → train-bc ×2 → compare → report works in 40 s. B beats A by +14.5 pts, p = 3.7e-9, the same numbers as phase 3.
- The clone and the 8 GB venv were deleted afterwards.
- To redo once the PM's docs land: they'll add `GITHUB_OWNER` to README, CHANGELOG and the other allowlisted docs, and the README steps must be run verbatim from the wheel (P4-7).

### R-real-3 ruling: **stay deferred**

QA used a movement-duration LDLJ (the jerk integral and duration over the span where speed ≥ 5% of peak), swapped into the production engine.

| Seed set | Profile | Flag changes | Binding bars | Bar values |
|---|---|---|---|---|
| dev | standard / low | 0/260 / 0/260 | pass → pass | identical |
| held-out | standard / low | 0/260 / 0/260 | pass → pass | identical |
| QA-private | standard / low | 0/260 / 0/260 | pass → pass | identical |

**But "unchanged" here means the change does almost nothing, not that it is shown to be safe.**
- On our data the 5%-of-peak span is almost always the whole episode:
  - it's shorter than the episode in only 2/160 (standard dev), 46/160 (low dev) and 1/150 (truncated real-like) episodes;
  - action noise keeps the speed above 5% of peak through the hold.
- On the truncated fixture, the short episodes' LDLJ z is **identical** before and after (median −1.90, min −6.9).
- So the bias R-real-3 targets (short real episodes at z −5 to −8) is neither reproduced nor fixed by this variant. Landing it would also end bit-for-bit parity with the phase 2 prototype, for no demonstrated gain.

Revisit only with:
- a duration normalization that measurably reduces |z| on short episodes in a variable-length fixture (e.g. `realify` truncated, cut in mid-motion);
- the bars unchanged on dev, held-out and private;
- the parity test updated deliberately.

### Still to review (superseded by the re-review below)

- The PM's P4-7 docs (README, CHANGELOG, CONTRIBUTING, SECURITY, RELEASE.md) and P4-8 launch drafts: not landed yet.
- The fixes for R1–R3.
- The rehearsal steps that depend on the docs.

## Re-review

Reviewed b355a98 (Coder: R1–R3) and b102d54 (PM: README, CHANGELOG, CONTRIBUTING, SECURITY, RELEASE.md, docs/launch/).

### R1–R3: verified

- **R1 (linking ignores the training filter): fixed.**
  - `link()` now returns `linked` and `full`. Only a checkpoint trained with `--keep all` is `full`.
  - For a filtered checkpoint, the card computes the subset's stats from `kept_episode_indices`. It says "learned from a filtered subset (keep = …, k of N episodes), of which x% failed", and it says so explicitly when the indices aren't recorded.
  - "trained on" / "training demos" appear only when every bc side is `full`. A filtered checkpoint never satisfies R5.
  - P4-5 has 3 new rows: filtered B with recorded indices (R2, "of which 0% failed"), filtered B without indices (R2, "filtered subset"), and R5 never for a filtered B (INCONCLUSIVE). **P4-5: 26/26.**
  - Mutation: `if keep == "all":` → `if True:` in `link()` fails all 3 new rows.
  - The Coder's `tests/test_p4_fix_linking.py` imports `bc_policy` from P4-5, so the helper emits `kept_episode_indices` only when a test passes them. Before that fix, my first version broke `test_old_filtered_checkpoint_without_indices`.
- **R2 (release must fail on the placeholder): fixed.**
  - `set_github_owner.py --check` exits 1 and lists the files while `GITHUB_OWNER` remains. The old preview is now `--dry-run`.
  - `release.yml` runs `--check` in the build job, and both publish jobs need `build`.
  - An invalid owner (`a--b`) is rejected with rc 2.
- **R3 (denylist leaks): fixed.**
  - The default is `~/.rrc_redactions.txt` (or `RRC_REDACTIONS`), and no `/home/claude` path is left.
  - A hit reports only "entry #k" and a location. A planted file-name hit printed 0 entries.
  - It reports how many history commits still contain an entry, without saying which (N1). Owners with consecutive hyphens are rejected (N2).
  - It never uploads.

### PM docs and launch drafts

The docs are honest overall:
- No third-party dataset or org name appears in any tracked file (release_check: 0 matches in 142 files).
- The real-data result is stated as "fixed and verified in simulation; the re-check on the real dataset is pending".
- The headline is framed as a checkpoint pair, and the data claim cites the 5-seed result.
- The launch drafts are marked DRAFT, and every `GITHUB_OWNER` goes through the tool.
- CHANGELOG, CONTRIBUTING and SECURITY are fine.

**RELEASE.md is correct and complete for a first-time publisher.** The steps are in order:
1. bundle pull
2. keep or squash history (release_check now tells Sebi that 2 commits contain a denylist entry)
3. set the owner and run `--check`
4. release_check plus the Mac check
5. private repo, then CI
6. the `pypi` environment with a required reviewer, and `testpypi`
7. pending trusted publishers
8. TestPyPI dry run plus an install test
9. tag, GitHub release, approve the deployment; twine fallback; yank / 0.1.1 guidance

I ran step 3 verbatim in a clone, and it works.

**QA reproduced the 5-seed numbers from the built wheel.** Same mix (seed 30000), train seeds 0–4, 200 eval episodes each:

| Train seed | A (all) | B (ok-and-success) | McNemar p |
|---|---|---|---|
| 0 | 0/200 | 29/200 | 3.7e-9 |
| 1 | 0/200 | 33/200 | 2.3e-10 |
| 2 | 2/200 | 39/200 | 1.5e-10 |
| 3 | 1/200 | 22/200 | 5.7e-6 |
| 4 | 0/200 | 15/200 | 6.1e-5 |

So "A 0–2/200, B 15–39/200, every p ≤ 6.1e-5" is true. But STATUS doesn't carry it (see D4).

### Required changes (PM; one-line doc edits)

- **D1. README "What we've measured": "At most 2% of good episodes are flagged" overclaims.**
  - Low-noise clean-only is flagged 2–5% (STATUS l.65, l.96).
  - Add "at standard noise (2–5% at low noise)". hf-forum.md already says "at standard noise".
- **D2. README "Statistics": "Wilson coverage … never below 0.91 for n = 20–200" drops the range it was measured on.**
  - The coverage was measured for true rates between 5% and 95% (phase3-code-review (e)). It is lower near 0 and 1.
  - Add "for success rates between 5% and 95%". STATUS l.74 needs the same qualifier.
- **D3. README "Report card": "A `bc:` checkpoint counts as trained on the scored dataset only when its dataset fingerprint matches" is stale after R1.**
  - Add: "…and it was trained on every episode (`--keep all`). A filtered checkpoint is reported as a subset of the scored dataset, and it never gets a clean verdict."
- **D4. STATUS doesn't carry the 5-seed per-seed ranges.** Phase 4 plan (P4-7) requires them there. README, hf-forum.md and x-thread.md cite "0–2/200 vs 15–39/200".
  - Add them to the STATUS headline line, citing this re-review.

Non-blocking: "**HARD** means an unusable recording" is stronger than the rule set supports. For example, a short episode or 20% saturation can still be usable. "a recording with a hard defect" would be accurate.

### P4-9 rehearsal, rerun: PASS

Run from a fresh `git clone` at b355a98, with temp owner `example-owner` in the clone only, and `RRC_REDACTIONS` set:
- **release_check: 9/9 PASS in 38 s.**
  - Wheel 6.1 MB; twine PASSED; Metadata 2.5, Apache-2.0, 3 license files.
  - Base venv 74 MB; quickstart 325 MB.
  - sdist suite: 239 passed, 29 skipped.
  - placeholder: none left.
  - redactions: 2 entries, 0 matches in 142 files, 2 history commits (Sebi's squash-or-keep call). No entry was printed.
- **README "Evaluate a policy" block, run verbatim from the built wheel.** The wheel was installed non-editable with `[score,sim,eval,lerobot]` into a clean Python 3.11 venv.
  - All 8 commands ran in 46 s (the README says 35–50 s). Weights 48b770… / 24d70c…, the same as P4-9.
  - The compare output matches the README excerpt exactly: 0/200 vs 29/200, Δ +14.5 [+9.9, +20.0], p = 3.7e-9.
  - The card gives R2. It says "A learned from all 90 episodes. B learned from a filtered subset (keep = ok-and-success, 59 of 90 episodes), of which 0% failed". "trained on" doesn't appear.
- **README "Record, tag and export" block** plus `rrc score --only-flagged --json-out … --overwrite` and `rrc record --list-groups`: all ran from the wheel. Tags are copied to `meta/rrc_tags.json`.
- Main repo: full suite 306 passed, 3 skipped in 142 s (DoD 11: < 150 s). ruff check and format are clean.
- The clone, wheel, venv and demo dirs were deleted afterwards.

### Final verdicts

| Subtask | Verdict |
|---|---|
| P4-1 packaging, hints, `set_github_owner.py`, `release_check.py` | APPROVED (R3 fixed: never uploads, no entry leaks) |
| P4-2 CI + release workflows | APPROVED (R2 fixed, with a static test) |
| P4-3 `rrc record --mix` | APPROVED |
| P4-4 `rrc report` | APPROVED (R1 fixed) |
| P4-5 rules test (QA) | 26/26, mutation-checked |
| P4-6 N3 + R-real-3 deferral | APPROVED |
| P4-7 docs | CHANGES REQUIRED: D1–D4 |
| P4-8 launch drafts | APPROVED (their numbers are reproduced above; they trace once D4 lands) |
| P4-9 rehearsal | PASS |

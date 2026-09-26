# Phase 1 Code Review

QA, 2026-09-24. Scope: commits 2731b1d..51f7594 (P1-0 to P1-5), checked against the acceptance criteria in `docs/phase1-plan.md` rev 2.
Environment: `/home/claude/.venvs/rrc` (lerobot 0.4.4, numpy 2.3.5, mujoco 3.14), plus a clean Python 3.11 venv built from the wheel with `[dev,sim]`.

## Summary

| Subtask | Verdict |
|---|---|
| P1-1 Package, CLI, session format, assets | **CHANGES REQUIRED** (R4) |
| P1-2 Env, policies, `rrc record` | **CHANGES REQUIRED** (R3) |
| P1-3 `rrc tag` / `rrc list` | **APPROVED** (R5 is a small fix, not a blocker) |
| P1-4 `rrc export` | **CHANGES REQUIRED** (R1 data loss, R2) |
| P1-5 e2e test | **APPROVED** (QA takes ownership of it as written) |
| P1-5 docs (Quickstart, STATUS) | **CHANGES REQUIRED** (R6, owner PM) |

The main path is solid. Every acceptance item passes. The blocking problems are all in failure and misuse paths, and one of them deletes the user's source data.

## What I ran

- `pytest`: **44 passed in 16.5 s**. `ruff check` and `ruff format --check` are clean.
- Clean venv from the built wheel, `[dev,sim]` only: **202 MB**. The wheel contains all 13 STL files, `LICENSE` and `SOURCE.md`. Record, tag and list work. Export fails cleanly with the install hint. The suite gives 36 passed and 2 skipped (lerobot modules skipped).
- Demo, run with `HF_HUB_OFFLINE` unset, in a temp dir:
  - `record` 10 episodes: 1.1 s, 10/10 success.
  - `tag`, `list`: correct.
  - `export`: 5.1 s.
  - Offline `LeRobotDataset` load: 10 episodes / 900 frames / 30 fps. The tag reached `rrc_tags.json`.
- Determinism across processes: two separate `record --seed 7 --noise 0.1` runs give byte-identical `.npz` arrays and identical `episodes.jsonl` apart from `recorded_at`.
- Acceptance bars (mine and STATUS agree):
  - scripted: 100%.
  - `--no-gravity-comp`: 70% on seed 0.
  - random: 0%.
  - stall and wrong: 0%.
  - 5 episodes record in well under 10 s.
  - Every episode has exactly `max_steps` frames.
- Divergence stress test: 400 episodes (random, random + noise 5, scripted + noise 5, wrong + noise 2) never diverged.

**Seeding deviation, approved.** `SeedSequence([seed, i]).spawn(2)` replaces the plan's `seed + i`. It fixes the neighbouring-seed overlap QA flagged earlier, it's covered by a test, and it's documented in `session.json`. The plan should be updated to match.

## Findings from trying to break it

| # | Input | Result |
|---|---|---|
| B1 | `rrc export S --out D --overwrite`, where `D` has `meta/info.json` and contains session `S` (for example `S` = `D`, or `S` = `D/runs/s`) | **Session deleted and export fails.** Output: `missing episode file ...`, then `D` no longer exists. The session is the tag source of truth, so the data can't be recovered. |
| B2 | Export fails partway (for example a missing `.npz`, or Ctrl-C) | The `rmtree` runs, then lerobot's `__del__` flushes its metadata buffer and recreates `<root>/meta/episodes/chunk-000/file-000.parquet`. The leftover root has no `info.json`. A re-export without `--overwrite` fails with "exists", and with `--overwrite` it's refused as "not a LeRobot dataset". A loader pointed at it goes to the Hub (G7). The user is stuck until they `rm` it by hand. |
| B3 | `record --noise nan` | Traceback (`Out of range float values are not JSON compliant`). It leaves a partial session with an orphan `episode_000000.npz` and no row. |
| B4 | `record --noise inf` | Traceback (`ValueError: non-finite action`). A divergence would raise `FloatingPointError`, which is also uncaught. |
| B5 | `record --out <existing file>` / `--out <path under a file>` | Traceback (`FileExistsError` / `NotADirectoryError`). |
| B6 | `record --out datasets/demo` (an exported dataset) | Succeeds: it writes `session.json`, `episodes.jsonl` and `episodes/` into the LeRobot root. That's how the B1 layout can come about by accident. |
| B7 | `tag --flag " padded "` then `--unflag " padded "` | The flag is stored stripped but not removed, because add strips whitespace and remove doesn't. |
| B8 | `export --out <symlink to dataset> --overwrite` | Traceback `OSError: Cannot call rmtree on a symbolic link`. Nothing is lost. |
| ok | Episode specs `1-`, `-`, `3-1`, `1,,2`, a huge index; bad label; conflicting flags; truncated `episodes.jsonl`; missing session; empty dir with `--overwrite`; a file with `--overwrite`; excluding every label | All give a clean `rrc: error:` and a non-zero exit, and nothing is modified. |

## Rulings on the Coder's deviations

**(a) Gravity compensation on by default (scripted = 100%). Keep it on.**
- `scripted` is the "good demo" generator and should give a clean positive class.
- The misses from sag in the uncompensated controller are smooth trajectories that fail by about 1 cm. For a phase 2 scorer that grades motion quality (jerk, stalls, length outliers), those are label noise, not junk data.
- The good/junk spread should come from mixing sources: `--noise` 0.05 / 0.1 / 0.25 (76% / 36% / 10%), `stall`, `wrong` and `random`. Keep `--no-gravity-comp` as the "near-miss" source.
- **Gap, needed before phase 2 (not blocking phase 1):** a session holds one policy and noise setting, and export works per session, so phase 1 can't produce *one* dataset that mixes good and junk episodes. Add `rrc record --append` (same env, fps and features; indices continue; per-episode `policy_params` is already recorded), or add a session merge.

**(b) Divergence aborts the recording instead of saving an `error` episode. Accepted for phase 1, on condition of R3.**
- It can't be reached with clipped position targets (0 of 400 stress episodes diverged).
- A NaN-filled "error" episode would also go on to poison `stats.json` on export.
- It must fail cleanly: no traceback, a clear message saying how many episodes were saved, and a session that stays valid (it already is, because rows are written only after each episode completes).
- `termination_reason="error"` stays reserved and unused. Note that in STATUS.

**(c) `--overwrite` deletes any directory that contains `meta/info.json`. Rejected in its current form (R1).** It can delete the session being exported (B1). It will also delete any LeRobot dataset the user pointed at by mistake, including one downloaded from the Hub that rrc never wrote.

## Required changes

1. **R1 (P1-4, blocker: data loss).** `src/robot_report_card/export/lerobot_writer.py:82-93` (`prepare_root`), called at :158.
   - Resolve both paths, and refuse if the session dir is inside `root` or the same as it, or if `root` is inside the session.
   - Only delete a root that rrc wrote: `meta/info.json` **and** `meta/rrc_tags.json` with `"tool": "robot-report-card"`. Anything else is refused with "delete it yourself".
   - Handle a symlinked `root` with a clean refusal (B8).
   - Add regression tests for B1 (both layouts) and for a foreign LeRobot dataset.
2. **R2 (P1-4).** `lerobot_writer.py:189-191`: release lerobot's writers before cleaning up. For example, keep `ds` in scope, call `ds.finalize()` inside a nested `try` (or `del ds; gc.collect()`), then `shutil.rmtree`. Add a test: a session with a deleted `.npz` must leave **no** `--out` directory, and a second export to the same `--out` must then succeed without `--overwrite`.
3. **R3 (P1-2).**
   - `src/robot_report_card/policies.py:125`: reject a noise value that isn't finite (`not math.isfinite(noise) or noise < 0`).
   - `src/robot_report_card/commands/record.py:67-70`: also catch `ValueError`, `FloatingPointError` and `OSError`, then `fail()` with "N episodes saved in <out>".
   - Tests for `nan`, `inf`, and an env stub that raises `FloatingPointError`.
4. **R4 (P1-1).** `src/robot_report_card/session.py:126-135` (`Session.create`): raise `SessionError` if `root` exists and is not a directory, or is a non-empty directory (B5, B6).
5. **R5 (P1-3, small).** `src/robot_report_card/tagging.py:58`: strip `remove_flags` the same way `add_flags` is stripped (B7).
6. **R6 (P1-5 docs, owner PM).**
   - README has no Quickstart (≤10 lines, required by P1-5).
   - STATUS only logs P1-2. Add P1-1/3/4/5, the deviations above (seeding, gravity compensation default, divergence abort, reserved `error`), the clean-venv size (202 MB), and the known limitations:
     - no locking, so two concurrent `rrc tag` runs can lose an update;
     - one policy per session (see ruling a);
     - `--repo-id` isn't validated.
   - Update plan §Seeding to `SeedSequence([seed, i])`.

## P1-5 end-to-end test (QA ownership)

`tests/test_p1_5_e2e.py` is **approved as written**, and QA takes it over. It runs the real CLI in subprocesses: record 3, tag, list (table and JSON), export, offline load. It checks counts, fps, `robot_type`, `codebase_version`, feature shapes, dtypes and names, sampled values against the `.npz`, the `task` string, the sidecar fingerprint, and that every `episodes.jsonl` field matches `rrc_tags.json`. It takes 5.7 s.

One non-blocking addition, which QA will make after R1–R5 land: call the installed `rrc` console script (via `shutil.which`) at least once instead of `python -m`, because DoD 2–4 are written against `rrc`.

The regression tests for B1–B7 belong with the owning subtasks (R1–R5), not in the e2e test.

## Re-review

After R1–R5 land, QA will rerun B1–B8, the full suite and the clean-venv Quickstart. R6 is reviewed separately as PM's docs.

## Re-review

QA, 2026-09-24. Scope: `b1ec05c` (R1–R5), `d5bdab3` (BrokenPipe), and the PM's uncommitted R6 edits (README, STATUS, plan rev 3).

**Suite:** 72 passed in 19.3 s (was 44). `ruff check` and `ruff format --check` are clean.

**README Quickstart, run verbatim** in a fresh temp copy of the repo:
- New `python3.11 -m venv`, then `pip install -e ".[dev,sim]"`: 22 s, **258 MB** with pip (README says about 200 MB, which is the uv figure).
- `record` / `tag` / `list` work. `export` without the extra gives a clean install hint and creates no `datasets/`.
- The `export` and `HF_HUB_OFFLINE=1` load lines, run in the lerobot venv with `HF_HUB_OFFLINE` unset for `export`: `10 900`.

**Break attempts, re-run:**

| Case | Before | Now |
|---|---|---|
| B1 `--overwrite` when the session is inside the root, is the root, or contains `--out` | session deleted | refused ("overlaps the session"), session intact. **R1 fixed.** |
| Foreign LeRobot dataset (no `rrc_tags.json`) with `--overwrite` | would be deleted | refused ("not a dataset exported by rrc"). An rrc-written dataset still overwrites fine. |
| B2 failure inside the lerobot write (a float64 `.npz` in episode 20 of 30; SIGINT at 3.6 / 4.0 / 4.4 / 4.8 s, traceback shows it inside `save_episode`) | leftover `meta/episodes/...parquet`, stuck | **no `--out` left behind**. Re-export to the same `--out` without `--overwrite` succeeds and loads (30 episodes). **R2 fixed.** |
| Missing `.npz` | half-written root | caught before lerobot is imported; clean error; nothing created |
| B3/B4 `--noise nan` / `inf` | traceback, orphan `.npz` | clean error, no directory created. **R3 fixed.** |
| B5 `--out` is a file / is under a file | traceback | clean errors |
| B6 `record --out` into an exported dataset | wrote into it | refused ("not empty"). **R4 fixed.** |
| B7 `--unflag " padded "` | flag not removed | removed. **R5 fixed.** |
| B8 `--overwrite` on a symlink | `OSError` traceback | clean refusal |
| BrokenPipe: `rrc list big \| head -1`, `list --json \| head -c 10` | — | exit 141, empty stderr |

**Non-blocking follow-ups (phase 2 backlog):**
1. A corrupt-but-present `.npz` (wrong dtype) makes `rrc export` print a `ValueError` traceback. Cleanup is still correct. `commands/export.py:49` should also catch `ValueError`.
2. `rrc record ... | head -1`: the `BrokenPipeError` from progress printing is caught by record's `OSError` handler. Recording stops with "N episodes saved" and exit 1 instead of 141. The session is valid and this matches SIGPIPE semantics. Either let `BrokenPipeError` reach `cli.main`, or keep recording with output silenced.
3. The README "~200 MB" should say "about 200–260 MB" (uv vs pip).

**Docs (R6):**
- README Quickstart is present, ≤10 command lines, and accurate.
- The plan rev 3 seeding, gravity-compensation and divergence text matches the code.
- STATUS is accurate as of the review *before* this one. At close-out the PM must:
  - move P1-1/2/4 from "In review" to Shipped;
  - update the test count to 72 and list `d5bdab3`;
  - mark R5 as done;
  - add follow-ups 1–3 above;
  - **commit** README, STATUS, plan rev 3 and `docs/reviews/` (all currently uncommitted).

This is bookkeeping, not a blocker.

### Final verdicts

| Subtask | Verdict |
|---|---|
| P1-1 Package, CLI, session format, assets | **APPROVED** |
| P1-2 Env, policies, `rrc record` | **APPROVED** |
| P1-3 `rrc tag` / `rrc list` | **APPROVED** |
| P1-4 `rrc export` | **APPROVED** |
| P1-5 e2e test (QA-owned) | **APPROVED** |
| P1-5 docs (R6) | **APPROVED**, once the STATUS close-out update is made and committed |

**PHASE 1 APPROVED.**

# Phase 1 Plan — Data Logger

Author: PM · Date: 2026-09-24 · Status: rev 3, QA changes applied (see end); rev 3 syncs seeding/gravity-comp with the build
Inputs: `README.md`, `STATUS.md`, `docs/research-brief.md` (Researcher, QA-reviewed),
`docs/spikes/sim_reach_to_lerobot.py` (proves sim → LeRobot round trip). Where the brief and this plan
disagree, the reconciled decisions D1–D5 in the QA review below are binding.

## Goal

A working vertical slice: generate SO-101 reach episodes in MuJoCo (CPU, state-only) with a scripted policy,
tag them, and export a LeRobot v3.0 dataset that `LeRobotDataset` opens offline. Out of scope for phase 1:
cameras, video, rendering, Hub push, gym-pusht, teleop, retagging an exported dataset in place.

## Definition of Done

1. `pip install -e ".[dev,sim,lerobot]"` works on Python ≥3.10 (tested 3.11) — **Linux, verified by QA**.
   macOS arm64 is verified by Sebi (see Needs Sebi's call).
2. `rrc record --env so101_reach --policy scripted --episodes 10 --seed 0 --out runs/demo` writes 10
   episodes of exactly 90 frames each to a session dir.
3. `rrc tag runs/demo --episode 3 --label fail --note "overshot"` persists to `runs/demo/episodes.jsonl`;
   `rrc list runs/demo` shows index, length, sim_success, final_error_m, label, notes.
4. `rrc export runs/demo --out datasets/demo --repo-id local/demo` writes a finalized LeRobot v3.0 dataset
   plus `meta/rrc_tags.json`.
5. A test loads it with `lerobot.datasets.lerobot_dataset.LeRobotDataset(repo_id, root=...)` under
   `HF_HUB_OFFLINE=1` and checks episode/frame counts, feature names/shapes/dtypes, fps, sampled values,
   and that `rrc_tags.json` matches the session.
6. `pytest` passes with no GPU and no network (vendored assets, `HF_HUB_OFFLINE=1`), whole suite < 2 min;
   record/tag tests pass without lerobot installed.
7. `STATUS.md` updated with the measured scripted success rate; each subtask reviewed by a different role.

## Design decisions

- **Two stages.** `record` writes our session format (one `.npz` per episode + `episodes.jsonl`); `export`
  converts to LeRobot. `tag` edits `episodes.jsonl` only (atomic temp-file + rename).
- **Pins/extras.** Core: `numpy<2.4` (required: numpy 2.4 crashes `save_episode` on shape-`(1,)` features).
  `sim` = `mujoco>=3.2`. `lerobot` = `lerobot==0.4.4`. `dev` = pytest, ruff. Only `export` imports lerobot.
- **Writer.** `LeRobotDataset.create(..., robot_type="so101_sim", use_videos=False)` → `add_frame` per frame
  with constant `task="Move the gripper tip to the target point."` → `save_episode` → **`finalize()`**.
  `root` must not exist; `--overwrite` deletes it deliberately.
- **Features (float32 unless noted):** `observation.state` (6), `observation.environment_state` (3, target
  xyz), `action` (6), `next.reward` (1,), `next.success` (1,) bool. State/action names `<joint>.pos` for
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. Session `.npz` stores all of these.
- **Env `so101_reach`** (brief §3): vendored `Simulation/SO101` MJCF + STL + Apache-2.0 LICENSE under
  `src/robot_report_card/sim/assets/so101/`; no `robot_descriptions`. fps=30, `m.opt.timestep = 1/(30*17)`
  (17 substeps/frame, asserted). Reset samples reachable target via FK of `q* ~ U(0.6·jnt_range)`, start at
  `qpos=0`. Fixed `--max-steps` (default 90), no early termination. `sim_success` = final-frame tip error < 2 cm.
- **Seeding (as built, QA-approved).** Episode i draws from `np.random.SeedSequence([seed, i])` (spawned into
  env and policy streams), so each episode is reproducible alone and different `--seed` runs never share
  episodes (the earlier `seed + i` overlapped neighbouring seeds). Recorded in `session.json`.
- **Policies.** Required: `scripted` (ramp to q*; `--noise` σ rad added to actions, 0 = smooth) and `random`.
  Nice to have: `stall`, `wrong`, `truncated`, randomized ramp speed.
- **Gravity compensation ON by default (QA ruling).** `scripted` adds a gravity offset (100% success at noise 0)
  so it is a clean "good demo" source; `--no-gravity-comp` (~70–80%) is the near-miss source. Junk comes
  from `--noise`, `stall`, `wrong`, `random`. A sim divergence aborts `record` cleanly (episodes saved so far
  stay valid) instead of saving an `error` episode; `termination_reason="error"` is reserved, unused.

## Tags and phase 2 hooks

Before export the source of truth is `episodes.jsonl` in the session. Per episode:
`episode_index, env_id, policy_name, policy_params (incl. noise), seed, fps, num_frames, duration_s,
sim_success, final_error_m, termination_reason (max_steps|truncated|error), label (success|fail|unlabeled),
notes, flags, recorded_at, rrc_version`. `sim_success` and `label` are never merged; `rrc tag --from-sim`
copies one into the other only when asked.

On export, write the snapshot `<root>/meta/rrc_tags.json` =
`{"schema_version":1, "tool":"robot-report-card", "rrc_version", "dataset":{codebase_version, total_episodes,
total_frames, episode_lengths}, "episodes":{"<exported idx>": {source_episode_index, ...all fields above}}}`.
Exported indices are renumbered when `--exclude-label fail` drops episodes, so `source_episode_index` is
mandatory. To retag: tag the session, re-export with `--overwrite`. Tags never go in `task`.
Why: `sim_success` + known `noise`/policy give phase 2 a labelled good/junk set to validate the scorer.

## Subtasks (ordered)

### P1-1 Package, CLI skeleton, session format, vendored assets
- **Scope:** `pyproject.toml` (src layout, core `numpy<2.4`, extras `dev`/`sim`/`lerobot`), `rrc` entry point
  with `record|tag|list|export` stubs and `--version`; `Episode`/`EpisodeMeta` dataclasses; session
  read/write; vendor SO101 assets + LICENSE as package data; `tests/conftest.py` sets `HF_HUB_OFFLINE=1`.
- **Acceptance:** `rrc --help` lists 4 subcommands; synthetic `Episode` round-trips through the session
  format bit-exactly; vendored MJCF loads with `mujoco.MjModel.from_xml_path` (skip if mujoco absent); ruff + pytest green.
- **Depends on:** nothing. **Does:** Coder. **Reviews:** QA.

### P1-2 `so101_reach` env, policies, `rrc record`
- **Scope:** `EnvAdapter` protocol (`reset(seed)`, `step(action)`, specs, `fps`, `tip_error()`) with the
  `so101_reach` implementation; `scripted` + `random` policies; `rrc record --env --policy --episodes --seed
  --noise --max-steps --out`. Controller may be improved (e.g. gravity offset); the 2 cm threshold may not be loosened.
- **Acceptance:** 5 episodes record in < 10 s; same seed ⇒ identical arrays; every episode has exactly
  `max-steps` frames; timestep assertion holds; `scripted --noise 0` succeeds on **≥60% of 50 seeded episodes**,
  `random` **≤10%**; measured rates written to STATUS.md; all metadata fields populated.
- **Depends on:** P1-1. **Does:** Coder. **Reviews:** QA (reruns the rates), Researcher (env matches brief §3).

### P1-3 `rrc tag` and `rrc list`
- **Scope:** set/clear `label`, set/append `notes`, add/remove `flags` per episode; bulk `--from-sim`;
  `list` as a table with `--json`.
- **Acceptance:** tags persist across processes; bad index/label → non-zero exit + clear message; atomic write
  (a simulated crash mid-write leaves the old file intact); runs without mujoco or lerobot installed.
- **Depends on:** P1-1 (parallel with P1-2 using synthetic sessions). **Does:** Coder. **Reviews:** QA.

### P1-4 `rrc export` → LeRobot v3.0 + `rrc_tags.json`
- **Scope:** `robot_report_card/export/lerobot_writer.py` implementing the writer and features above;
  `meta/rrc_tags.json` snapshot; `--overwrite`; `--exclude-label fail`. Clear error if lerobot isn't installed.
- **Acceptance (tests use `pytest.importorskip("lerobot")`, offline):** dataset loads; counts, feature
  names/shapes/dtypes, fps match session; a sampled frame equals the `.npz` (float32 tolerance); sidecar
  fingerprint matches `info.json`; with `--exclude-label fail` indices are contiguous and `source_episode_index` correct;
  exporting to an existing root without `--overwrite` fails cleanly.
- **Depends on:** P1-1; P1-2 for the real-sim test (core logic testable on synthetic sessions).
  **Does:** Coder. **Reviews:** Researcher (format), QA.

### P1-5 End-to-end test, Quickstart, STATUS
- **Scope:** one e2e pytest: record 3 → tag → export → load offline. README Quickstart (≤10 lines). STATUS entry.
- **Acceptance:** QA runs the Quickstart in a clean venv with `.[dev,sim]` (record/tag/list) and in the
  existing `/home/claude/.venvs/rrc` for the lerobot extra (export + load); every DoD item passes; failures go
  back to the owning subtask, not patched silently.
- **Depends on:** P1-2, P1-3, P1-4. **Does:** QA (test + checklist), PM (docs/STATUS).
  **Reviews:** Coder reviews QA's test; QA reviews PM's docs.

## Risks

| Risk | Mitigation |
|---|---|
| LeRobot format/API churn (0.3 wrote v2.1, 0.4 writes v3.0) | Pin `lerobot==0.4.4`; use its writer; round-trip test is the gate. |
| numpy 2.4 breaks `(1,)` features (G1) | `numpy<2.4` in core deps; revisit on lerobot bump. |
| Unfinalized/missing root silently hits the Hub | Always `finalize()`; `HF_HUB_OFFLINE=1` in tests; check `info.json` exists before loading. |
| Heavy lerobot install (8.1 GB on Linux with CUDA torch; Mac unmeasured) | Optional extra; only export imports it; record/tag tests run without it. |
| Scripted reach only ~75% at noise 0 (gravity sag) | Resolved: gravity offset on by default → 100%; threshold still 2 cm. |
| Mac install unverified by crew | Sebi runs Quickstart on the MacBook Air (below). |

## Defaults (Sebi can override at milestone)

1. **`lerobot` (with torch) is an optional extra** used only by export, instead of hand-writing the format.
   Install weight: ~8.1 GB / ~2.5 min on Linux (CUDA torch); expected much smaller on Apple Silicon, unmeasured.
2. **Hugging Face Hub upload is out of scope for phase 1** — local datasets only.

## Needs Sebi's call

1. Run the README Quickstart on the MacBook Air and report: install succeeded?, install size and time,
   and whether `rrc export` + load works. This is the only macOS verification phase 1 gets.

## QA Review (2026-09-24)

**Verdict: APPROVED WITH CHANGES.** The structure is right: two stages, lerobot's own writer, lerobot kept as an optional extra, state-only. The plan was written before the brief existed, so it has to be brought in line with it. A few DoD items can't be tested as written. The Coder should follow the reconciled decisions below. They override the plan and the brief wherever the two disagree.

**Findings:**
1. **DoD 1 and 2 break each other.** DoD 1 installs `.[dev,lerobot]`, but `record` needs the `sim` extra (mujoco), so DoD 2 fails after the documented install.
2. **Sim task.** The plan assumes grasp/lift and a cube. The brief picks an SO-101 **reach** task (no grasp, no object), and QA reproduced it (86% scripted success). **The ≥80% bar fails.** With one RNG per seed (`default_rng(seed)`, how `--seed` will work), `smooth` succeeds on 14/20 on seeds 0–19 and 77/100 over 100 seeds. The brief's 85% came from a single shared RNG. The timestep fix in D2 doesn't change these numbers (QA checked).
3. **There are three different names for the tags sidecar:** plan `meta/rrc_episodes.jsonl`, brief `meta/rrc_tags.json`, spike `meta/rrc_tags.jsonl`. The schemas also differ: the plan keeps `sim_success` and `label` separate, the brief uses one `success` bool.
4. **"Plus any fields LeRobot's per-episode metadata can hold."** The brief shows `save_episode` accepts no custom metadata and rejects extra episode columns. The sidecar alone is the answer.
5. **Features.** The plan says state + action only. The brief and spike also write `observation.environment_state` (target xyz), `next.reward` and `next.success`. Those `(1,)` features are exactly what triggers the numpy 2.4 crash (QA reproduced it), so the `numpy<2.4` pin is required, not optional.
6. **DoD 6 (no network) can't be met with `robot_descriptions`.** It git-clones 295 MB at import time. Also, a `LeRobotDataset` load on a missing or unfinalized root silently falls back to the Hub.
7. **DoD 1 "macOS arm64" can't be verified by the crew** (no Mac, and the brief marks the Mac install as unverified). The Mac check has to come from Sebi.
8. **The "clean venv" in P1-5 costs 8.1 GB and about 2.5 min on Linux** (CUDA torch, CPU index blocked by the proxy). Needs Sebi's call #1 says "~1 GB", which is wrong for Linux and unmeasured for Mac.
9. **Export with fail episodes excluded renumbers `episode_index`,** so the sidecar has to record the source session index.
10. **`termination_reason` isn't defined for a fixed-length reach task,** and neither is when `sim_success` is evaluated.
11. **The stretch goals (`--cameras`, `--render`) can't be tested here.** There's no OpenGL in the container (`mujoco.Renderer` failed), and on macOS `mujoco.viewer` needs `mjpython`.
12. **`.gitignore`.** It ignores every dir named `data/` (so a `src/robot_report_card/data/` module would silently not be committed). It also doesn't ignore `runs/` or `datasets/`.
13. **Scope** fits one session: the spike already proves env + export, and tag/list is just JSON. The DoD is testable once findings 1, 6 and 7 are fixed.

**Required changes:**
- R1 (finding 1): DoD 1 becomes `pip install -e ".[dev,sim,lerobot]"`. `sim` = `mujoco>=3.2` (vendored assets, no `robot_descriptions`). `lerobot` = `lerobot==0.4.4`. Put `numpy<2.4` in **core** dependencies.
- R2 (findings 2 and 10): the env is `so101_reach` as specified in brief §3, with a fixed `--max-steps` (default 90) and no early termination. `sim_success` = final-frame tip error < 2 cm. `termination_reason` ∈ `max_steps|truncated|error`. Policies: `scripted` with `--noise` (0 means smooth) and `random` are required. `stall` / `wrong` / `truncated` / random speed are nice to have. P1-2 bar: ≥60% success over 50 seeded episodes at noise=0, `random` ≤10%, and the measured rate recorded in STATUS.md. The Coder may improve the controller (e.g. a gravity offset) but must not loosen the 2 cm threshold.
- R3 (findings 3 and 4): see decision D3 below. Delete the "per-episode metadata" clause.
- R4 (finding 6): vendor the SO-101 MJCF + STL + LICENSE into the package. `tests/conftest.py` sets `HF_HUB_OFFLINE=1`. Export tests use `pytest.importorskip("lerobot")`.
- R5 (findings 7 and 8): DoD 1 = Linux verified by QA. Add to Needs Sebi's call: "run the Quickstart on the MacBook Air and report the install size and time." P1-5 may reuse `/home/claude/.venvs/rrc` for the lerobot extra, and the clean-venv check applies to `[dev,sim]` only. Correct the "~1 GB" figure.
- R6 (finding 11): drop the phase 1 stretch goals.
- R7 (finding 12): fix `.gitignore` (`/data/`, `runs/`, `datasets/`).

**Reconciled decisions (binding for the Coder):**
- D1 **Format / pins:** LeRobot v3.0 via `lerobot==0.4.4` writer (`create` → `add_frame` per frame with a constant `task` string → `save_episode` → **`finalize`**). `numpy<2.4`. Python ≥3.10, tested on 3.11. Only `export` imports lerobot.
- D2 **Features:** `observation.state` (6), `observation.environment_state` (3), `action` (6), float32. `next.reward` (1,) float32. `next.success` (1,) bool. Names are `<joint>.pos` as in the brief. fps=30 with `m.opt.timestep = 1/(30*17)` so the physics lines up with the timestamps. `robot_type="so101_sim"`. The session `.npz` stores all of these.
- D3 **Tags:** before export, the source of truth is the session's `episodes.jsonl`, which `rrc tag` edits with an atomic write. On export, write **`<root>/meta/rrc_tags.json`** = `{"schema_version":1, "tool", "rrc_version", "dataset":{codebase_version,total_episodes,total_frames,episode_lengths}, "episodes":{"<exported idx>": {source_episode_index, sim_success, label, notes, flags, policy_name, policy_params, seed, termination_reason, final_error_m, recorded_at, ...all plan fields}}}`. It's a snapshot. To retag, tag the session and re-export with `--overwrite`. Retagging an exported dataset in place is phase 2. Never put tags in `task`.
- D4 **Tests:** offline (`HF_HUB_OFFLINE=1`, vendored assets), suite under 2 min (lerobot import is about 3 s, the sim runs at 70 eps/s). Record/tag tests must pass without lerobot installed.
- D5 **Out of scope:** cameras, video, rendering, Hub push, gym-pusht.

### Changes applied (PM, 2026-09-24, rev 2)
- R1: DoD 1 installs `.[dev,sim,lerobot]`; `sim`=`mujoco>=3.2`, `lerobot`=`lerobot==0.4.4`, `numpy<2.4` in core.
- R2: env = `so101_reach`, fixed 90 frames, no early termination, success = final tip error < 2 cm,
  `termination_reason` ∈ `max_steps|truncated|error`; bars ≥60%/50 seeds (noise 0) and random ≤10%, rate logged in STATUS.
- R3/D3: one sidecar `meta/rrc_tags.json` snapshot (schema per D3, `source_episode_index`); session
  `episodes.jsonl` is the source of truth pre-export; "per-episode LeRobot metadata" clause removed.
- R4: vendored SO101 MJCF/STL/LICENSE, no `robot_descriptions`; `HF_HUB_OFFLINE=1` in conftest; `importorskip("lerobot")`.
- R5: DoD 1 = Linux verified by QA; Mac check moved to Needs Sebi's call; clean venv only for `[dev,sim]`; "~1 GB" corrected.
- R6: stretch goals (cameras, render) removed. R7: `.gitignore` now `/data/`, `runs/`, `datasets/`.
- D1/D2/D4/D5 folded into Design decisions; earlier open decisions recorded as defaults Sebi can override.

### QA Re-review (2026-09-24)
**Verdict: APPROVED.** R1–R7 are applied and the plan agrees with the brief on the sim, tags file, pins, features, seeding, asset path and scope. I checked `.gitignore` in a scratch git repo: `src/robot_report_card/data/` and `sim/assets/**` are tracked, and `/data/`, `runs/`, `datasets/` are ignored. Non-blocking: `default_rng(seed + episode_index)` means runs with neighbouring `--seed` values share episodes (seed 0, episode 1 = seed 1, episode 0). That's fine for phase 1, but P1-2 should document it.

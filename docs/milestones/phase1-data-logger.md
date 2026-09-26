# Milestone: Phase 1 — Data Logger

PM, 2026-09-24. Status: **built and approved by QA. Phase 2 waits for Sebi's go-ahead.**
Details: `STATUS.md` (Session 1), `docs/phase1-plan.md` (rev 3), `docs/reviews/phase1-code-review.md`.

## What was built

A command-line tool, `rrc`, that runs on a laptop CPU with no GPU and no network:

- `rrc record`: runs a simulated SO-101 arm (MuJoCo) doing a "reach the target point" task. It saves
  fixed 3-second episodes (90 frames at 30 fps) to a local session folder.
  - Policies: `scripted` (good demos), `random`, `stall`, `wrong`, plus `--noise` for shaky versions.
  - The same seed always gives the same episodes.
- `rrc tag` / `rrc list`: mark episodes success/fail/unlabeled, and add notes and flags. The simulator's own
  success verdict is stored separately and is never overwritten by a human label.
- `rrc export`: writes a standard LeRobot v3.0 dataset. It uses LeRobot's own writer, pinned to
  `lerobot==0.4.4`. Our tags and per-episode details go into a side file, `meta/rrc_tags.json`.
- A README Quickstart (7 commands) that QA ran word for word in a fresh environment.

**Proof:** 72 tests pass in about 19 s. QA's end-to-end test records, tags, exports and reloads the data with
LeRobot's own loader. QA also tried eight ways to break it (bad paths, overwriting the wrong folder,
Ctrl-C mid-export, NaN noise, and so on); all now fail with a clear message and no data is lost. The basic
install (`[dev,sim]`) is about 200–260 MB.

**Measured success, 2 cm threshold, seeds 0–9:**
- scripted: 100%
- scripted without gravity compensation: 81%
- noise 0.05 / 0.1 / 0.25: 71% / 34% / 8%
- random: 0.2%
- stall and wrong: 0%

**Reviewed by:** QA reviewed every subtask. The Researcher reviewed the environment and the data format. QA reviewed the PM's docs.

## Deviations from the plan, and why

1. **Seeding.** Episode `i` uses `SeedSequence([seed, i])` instead of `seed + i`. With the old scheme,
   `--seed 0` and `--seed 1` shared most of their episodes.
2. **Gravity compensation is on by default,** so `scripted` succeeds 100% instead of about 75%. The misses came from
   the arm sagging about 1 cm short, which is label noise rather than useful "bad data". `--no-gravity-comp`
   still produces those near-misses when we want them.
3. **A sim crash stops `record` cleanly instead of saving a broken episode.** A NaN-filled episode would
   corrupt the dataset statistics. Episodes saved before the crash stay valid, and none of 400 stress episodes crashed.
4. **Plan rev 2 changed the task from grasping to reaching,** following the research brief. Reaching needs no object and no grasp
   controller, and it matches the real SO-101 joint names.

## Open risks

- **The Mac is untested.** The crew has no Mac. The install size and time of the LeRobot extra on Apple Silicon are unknown
  (on Linux it's about 8 GB, because of the CUDA torch build).
- **LeRobot changes its format between minor versions.** We're pinned to 0.4.4 and numpy <2.4 (numpy 2.4
  crashes LeRobot's writer). Upgrading needs the round-trip test to pass again.
- **Small known limits:**
  - `rrc tag` has no file locking.
  - `--repo-id` isn't validated.
  - There are two cosmetic CLI follow-ups in the STATUS backlog.

## Decisions for Sebi

1. **Run the Quickstart on your MacBook Air** and tell us: did it install, how big and how long for
   `[dev,sim]` and `[dev,sim,lerobot]`, and do export + load work? This is phase 1's only macOS check.
2. **License: keep MIT or switch to Apache-2.0?** → **Sebi: Apache-2.0 (done).** LeRobot and the SO-101 model we ship are Apache-2.0.
   MIT works (we include their license file); Apache-2.0 matches the ecosystem and adds a patent grant.
3. **Defaults we applied (say if you want them changed):** → **Sebi: keep both.** LeRobot is an optional install used
   only for export; Hugging Face Hub upload is not in phase 1.
4. **Approve the start of phase 2** as proposed below, starting with `--append`/merge.

## Proposed phase 2: dataset scoring

**Goal:** `rrc score <dataset>` gives each episode a quality score with reasons: jerky motion, stalls,
failed attempts, length outliers. It works on both our exported datasets and real SO-101 LeRobot datasets.

**First prerequisites (before any scoring code):**
1. **Mixed datasets.** Today a session holds one policy and one noise setting, so we can't export a single dataset
   that mixes good and junk episodes. Add `rrc record --append` (or `rrc merge` for sessions).
   Per-episode policy details are already recorded, so this is small.
2. **A labelled benchmark dataset** built with (1): scripted episodes plus noisy, stalled, wrong, random
   and near-miss episodes, with the known answer for each episode. We validate the scorer against it.
3. **Units.** The sim records radians, while real SO-101 data uses degrees and a 0–100 gripper value. Metrics must
   either handle units or normalize with the dataset's `stats.json`.

Scoring design, thresholds and the report format get planned only after Sebi approves.

## Next step

**The crew waits for Sebi's go-ahead.** No phase 2 work starts until then.

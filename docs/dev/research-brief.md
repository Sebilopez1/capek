# Research Brief: Phase 1 Data Logger

Researcher, 2026-09-24. Ground truth is the installed `lerobot==0.4.4` source (the latest on PyPI today) plus experiments run in this container (Linux, Python 3.11, CPU only).
Runnable proof: `HF_HUB_OFFLINE=1 python docs/spikes/sim_reach_to_lerobot.py <new_dir>` goes from the vendored SO-101 model through a LeRobot dataset plus `meta/rrc_tags.json` and reloads it, offline, in about 5.5 s. Add `--measure` to print success rates.

## Recommendations

1. **Target format: LeRobot dataset v3.0** (`codebase_version: "v3.0"`, which is what lerobot 0.4.x writes). Don't write v2.1: lerobot 0.4 only loads it through a conversion script.
2. **Write the files through `lerobot` itself (`LeRobotDataset.create/add_frame/save_episode/finalize`). Don't hand-roll the format.** Keep it behind one module (`robot_report_card/export/lerobot_writer.py`) and make it an optional extra (`pip install robot-report-card[lerobot]`). The recorder and tagger stay lerobot-free, so they stay fast and light to test.
   - Pin `lerobot==0.4.4` and **`numpy<2.4`**. See gotcha G1: numpy 2.4 breaks shape-`(1,)` features in lerobot 0.4.4.
   - Why not hand-roll: v3.0 packs many episodes into each parquet file, keeps a wide episodes table (7 + 10 × number of features columns: 109 for our 5 features, with per-feature min/max/mean/std/count/q01/q10/q50/q90/q99) and embeds HF `datasets` schema metadata in the parquet footers. Copying all that means chasing a moving target. lerobot *is* the spec.
3. **Simulator: a small custom SO-101 "reach" env** built on `mujoco` plus the official SO-101 MJCF and meshes from TheRobotStudio's SO-ARM100 repo (Apache-2.0), **vendored at `src/robot_report_card/sim/assets/so101/`** (16 MB, so no `robot_descriptions` or git clone at runtime). It is state-only, needs no rendering, and has 6 joints named exactly like real SO-101 LeRobot data. `sim_success` means the gripper tip is within 2 cm of the target on the final frame. Verified here: about 70 episodes/s on CPU.
   - Runner-up and second env later: `gym-pusht` (official HF, has `is_success` built in) with `pymunk<7`.
4. **Tags: the session's `episodes.jsonl` is the source of truth. `rrc export` writes a snapshot to `<root>/meta/rrc_tags.json`** (`schema_version`, dataset fingerprint, episodes keyed by exported index, with `sim_success` and the human `label` as separate fields). Also record per-frame `next.success` / `next.reward` features from the sim. **Never** put tags in the task string. See §4.
5. Dev venv left at **`/home/claude/.venvs/rrc`** (lerobot 0.4.4, numpy 2.3.5, mujoco 3.14.0, torch 2.10 CUDA build; `robot_descriptions` 3.2.0 is still installed but no longer needed). It takes 8.1 GB because of the CUDA torch wheels (G4). Other test venvs were deleted.

## 1. LeRobot v3.0 on-disk format (verified from source and a written dataset)

```
<root>/
  meta/info.json                       # schema + counters (below)
  meta/stats.json                      # dataset-level stats per feature: min,max,mean,std,count,q01,q10,q50,q90,q99
  meta/tasks.parquet                   # index = task string, column task_index
  meta/episodes/chunk-000/file-000.parquet   # one row per episode (see columns)
  data/chunk-000/file-000.parquet      # frames of MANY episodes concatenated (new file after ~100 MB)
  videos/<video_key>/chunk-000/file-000.mp4  # only if dtype=="video"; many episodes concatenated per mp4
```
- **info.json**: `codebase_version, robot_type, total_episodes, total_frames, total_tasks, chunks_size (1000), data_files_size_in_mb (100), video_files_size_in_mb (200), fps, splits {"train":"0:N"}, data_path "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", video_path (null when no videos), features`.
- **features**: `{name: {"dtype": "float32"|"int64"|"bool"|"image"|"video"|"string", "shape": [..], "names": [..]|null}}`. Names must not contain `/`. lerobot auto-adds `timestamp` (float32), `frame_index`, `episode_index`, `index` (global frame index) and `task_index` (all int64).
- **Data parquet columns**: 1-D vectors are stored as `fixed_size_list<float>[N]`, and shape `(1,)` is stored as a scalar column. `timestamp = frame_index / fps` unless you pass one. It must stay on the fps grid (load-time tolerance is `tolerance_s=1e-4`).
- **Episodes parquet columns**: `episode_index, tasks (list<str>), length, data/chunk_index, data/file_index, dataset_from_index, dataset_to_index, meta/episodes/chunk_index, meta/episodes/file_index, stats/<feature>/<stat>`. With videos it also has `videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}`.
- **Video vs image**: `dtype:"video"` frames are written as temporary PNGs and then encoded to MP4 (default codec `libsvtav1`, needs ffmpeg/PyAV). `dtype:"image"` embeds PNG bytes in the parquet. **Phase 1 is state-only, so no video, `use_videos=False`, and no ffmpeg/torchcodec touched** (torchcodec is only imported lazily for video decoding).
- Convention used by lerobot's own sim/RL code: `observation.state`, `observation.environment_state`, `observation.images.<cam>`, `action`, `next.reward`, `next.done`, `next.success`.
- v2.1 (for reading older Hub datasets only) used one parquet per episode plus `meta/episodes.jsonl`, `episodes_stats.jsonl` and `tasks.jsonl`.

## 2. Write path (verified round trip, lerobot 0.4.4)

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NAMES = [f"{j}.pos" for j in J]                      # same naming as real SO-101 datasets
features = {
  "observation.state":             {"dtype": "float32", "shape": (6,), "names": NAMES},
  "observation.environment_state": {"dtype": "float32", "shape": (3,), "names": ["target_x", "target_y", "target_z"]},
  "action":                        {"dtype": "float32", "shape": (6,), "names": NAMES},
  "next.reward":                   {"dtype": "float32", "shape": (1,), "names": None},
  "next.success":                  {"dtype": "bool",    "shape": (1,), "names": None},
}
ds = LeRobotDataset.create(repo_id="rrc/sim_reach", fps=30, features=features,
                           root="outputs/sim_reach", robot_type="so101_sim", use_videos=False)
for ep in episodes:
    for f in ep.frames:
        ds.add_frame({"observation.state": f.state.astype(np.float32),          # exact dtype + shape required
                      "observation.environment_state": f.target.astype(np.float32),
                      "action": f.action.astype(np.float32),
                      "next.reward": np.array([f.reward], np.float32),          # shape (1,) ndarray, not a scalar
                      "next.success": np.array([f.success]),
                      "task": "Move the gripper tip to the target point."})    # REQUIRED on every frame
    ds.save_episode()            # writes parquet rows + episode metadata + stats
ds.finalize()                    # REQUIRED: closes parquet writers (footer), else the dataset won't load
r = LeRobotDataset("rrc/sim_reach", root="outputs/sim_reach")   # local read if root is complete; see G7
r.num_episodes, r.num_frames, r[0]["observation.state"]          # torch tensors; r.meta.episodes is an HF Dataset
```
Signatures (0.4.4): `create(repo_id, fps, features, root=None, robot_type=None, use_videos=True, tolerance_s=1e-4, image_writer_processes=0, image_writer_threads=0, video_backend=None, batch_encoding_size=1, vcodec="libsvtav1", metadata_buffer_size=10, streaming_encoding=False, ...)`, `add_frame(frame: dict)`, `save_episode(episode_data=None, parallel_encoding=True)`, `finalize()`. `save_episode` takes **no** custom metadata argument.

Round trip result: 3 episodes / 65 frames, then 4 SO-101 sim episodes / 360 frames. Both reloaded with correct counts, keys, tasks and per-frame `task` string. Write+read takes about 1 s after a roughly 3–8 s `import lerobot`.

**Gotchas (all hit or confirmed here):**
- **G1 (blocker): numpy 2.4.x plus any shape-`(1,)` feature crashes `save_episode`** with `TypeError: only 0-dimensional arrays can be converted to Python scalars`. lerobot's own `gym_manipulator.py` uses the same pattern, so this is an upstream bug. Pin `numpy<2.4` (2.3.5 works).
- G2: `root` must **not** exist (`mkdir(exist_ok=False)`). Pick a fresh dir or delete deliberately. Resuming = open with `LeRobotDataset(repo_id, root=...)`.
- G3: dtype and shape are validated exactly (`float64` state is rejected). Extra or missing keys raise errors.
- G4: install weight. `pip install lerobot` pulls torch, torchvision, diffusers, wandb, rerun-sdk, datasets, opencv, av, pynput and others (113 packages). On Linux the default torch wheel is the CUDA build: **8.1 GB venv, about 2.5 min install**. In CI, install torch from the CPU index first (`--index-url https://download.pytorch.org/whl/cpu`, about 1–1.5 GB total). *That index was blocked by this container's proxy, so this step is unverified.* On an Apple Silicon Mac the torch wheels are CPU/MPS only, so the install should be much smaller. **Unverified: no Mac here.** Needs Python ≥3.10. I tested 3.11, which is the recommended version.
- G5: lerobot's own edit tools (`delete_episodes`, `split_dataset`, `merge_datasets`) rebuild `meta/` and **drop both extra episode columns and our sidecar** (verified). Our tools must re-map tags themselves.
- G6: lerobot minor versions change the format (0.3 wrote v2.1, 0.4 writes v3.0). Keep the pin, and add one CI test that writes a dataset and reloads it with lerobot.
- **G7: loading a missing or half-written root silently goes to the Hub.** `LeRobotDataset(repo_id, root=...)` catches `FileNotFoundError`, creates `<root>/meta/` and calls `pull_from_repo` (verified here: with `HF_HUB_OFFLINE=1` a nonexistent root raises `OfflineModeIsEnabled` and leaves an empty `meta/` behind). Loading without `finalize()` does the same. Set `HF_HUB_OFFLINE=1` in tests and CI. Our CLI should check that `<root>/meta/info.json` exists before calling lerobot.
- **G8: the physics timestep must divide the frame period.** The MJCF default of 0.002 s × 17 substeps = 0.034 s ≠ 1/30 s, so the sim runs about 2% off the stamped timestamps. Set `m.opt.timestep = 1/(30*17)` after loading and assert `substeps * timestep == 1/fps`. The spike does this.

## 3. Simulator choice

| Option | Install (tested) | State-only, no GL? | Success signal | SO-arm fit | Verdict |
|---|---|---|---|---|---|
| **Custom reach on `mujoco` + vendored SO-101 MJCF** | `pip install mujoco`, ~160 MB venv; works | yes, 10k physics steps in 0.11 s | we define it (tip-to-target < 2 cm) | exact: 6 joints, same names/order | **Pick** |
| `gym-pusht` 0.1.6 | works **only with `pymunk<7`** (7.x: `no attribute add_collision_handler`) | yes; rgb render also works headless | `info["is_success"]` (coverage > 95%), reward = coverage | 2-D pusher, not an arm | Runner-up |
| `gym_soarm` 0.4.0 (3rd party, SO-ARM100 pick-place) | installs; pins `mujoco<3` / dm-control 1.0.14 | **no**: renders 4 cams at 640x480 every step even for `obs_type="state"`; hung under EGL here | `is_success` | good | Reject |
| `gym-so100` 0.1.2 (3rd party) | dm-control based | **no**: `obs_type="state"` raises NotImplementedError | reward 0–4 | bimanual SO100 | Reject |
| `gym-aloha` / `gym-hil` (official) | MuJoCo + dm-control / pygame+hidapi | aloha needs pixels for typical use | yes | 14-DoF bimanual / Franka | Too heavy / off-target |

**Env spec for the Coder** (prototype in the spike):
- Model: `mujoco.MjModel.from_xml_path("src/robot_report_card/sim/assets/so101/so101_new_calib.xml")` (load it via `importlib.resources` in the package; `scene.xml` adds a floor and light). It has 6 hinge joints/actuators named `"1"`..`"6"` (position actuators, kp=17.8) and a site named `"gripper"`. The MJCF timestep is 0.002 s: override it to `1/(30*17)` for 17 `mj_step`s per 30 fps frame (G8).
- Reset: sample `q* ~ U(0.6·jnt_range)`, then target = FK(q*) (always reachable). Start from `qpos = 0`. State = `qpos` (6), env state = target xyz, action = joint position target (6). Episode length 90 frames (3 s).
- Seeding: one `np.random.default_rng(seed)` per episode (seed = base_seed + episode_index), so any episode can be reproduced on its own. Rates below are from `--measure`: seeds 0–99, one RNG per seed, corrected timestep. My first figures (85% smooth, 0% jerky) came from one RNG shared across episodes, and those overstated the scripted policy.
  - `smooth`: linear ramp to q* over 2 s. **77% success** (seeds 0–19: 70%), median error 1.4 cm. The misses come from gravity sag, which gives useful natural variance. A gravity-offset controller could raise the rate, but don't loosen the 2 cm threshold.
  - `jerky`: smooth plus N(0, 0.25 rad) action noise. 8% success, 6.8 cm error.
  - `stall`: freeze the action after 1 s. 0%, 14.5 cm.
  - `wrong`: head to −q*. 0% (measured with the shared RNG only), about 52 cm.
  - Also add randomized speed (1–3 s ramps), `random` (uniform actions) and `truncated` (stop early, short length outlier). Together these give phase 2 scoring clear labels for jerk, stalls, failures and length outliers.
- Rendering (optional, later): `mujoco.Renderer(m, H, W)`. It **failed here** (no OpenGL in the container). CI would need `MUJOCO_GL=egl` or `osmesa` plus system libs. On a Mac the default backend should work, but that is **unverified**. Keep images out of phase 1.
- Assets are **vendored** (done): `src/robot_report_card/sim/assets/so101/` holds `so101_new_calib.xml`, `scene.xml`, `assets/*.stl` (13 meshes; the `.part` files were dropped), the upstream Apache-2.0 `LICENSE` and a `SOURCE.md` recording upstream commit `63eede5`. Total 16 MB. It loads standalone offline. `robot_descriptions` is **not** a runtime dependency, because its import git-clones the full 295 MB repo and needs GitHub. Package data must include `sim/assets/**`.
- Units: the sim uses radians. Real SO-101 LeRobot recordings use degrees (arm) and 0–100 (gripper). Phase 2 metrics should be unit-aware or normalize per-feature with `stats.json`.

## 4. Where tags live

**Two stages, one schema (aligned with `docs/phase1-plan.md` D3):**
- **Before export, the source of truth is the session's `episodes.jsonl`**, one row per recorded episode: `episode_index, env_id, policy_name, policy_params, seed, fps, num_frames, duration_s, sim_success, final_error_m, termination_reason (max_steps|truncated|error), label (success|fail|unlabeled), notes, flags, recorded_at, rrc_version`. `rrc tag` edits only this file, writing to a temp file and renaming it. **`sim_success` (from the sim, never edited) and `label` (from a human) are separate fields**, and they are only copied across when the user asks (`--from-sim`).
- **On export, write the snapshot `<root>/meta/rrc_tags.json`.** `push_to_hub` uploads everything except `images/`, so it travels with the dataset, and the lerobot loaders ignore it (verified). To retag, tag the session and re-export with `--overwrite`. Editing an exported dataset in place is phase 2. The spike writes exactly this shape:
```json
{"schema_version": 1, "tool": "robot-report-card", "rrc_version": "0.0.0-spike",
 "dataset": {"codebase_version": "v3.0", "total_episodes": 4, "total_frames": 360, "episode_lengths": [90,90,90,90]},
 "episodes": {"0": {"source_episode_index": 0, "sim_success": true,  "label": "unlabeled", "final_error_m": 0.0061, "notes": "", "flags": [],
                    "policy_name": "scripted", "policy_params": {"style": "smooth"}, "seed": 0, "termination_reason": "max_steps", "...": "..."},
              "1": {"source_episode_index": 1, "sim_success": false, "label": "fail", "notes": "looked shaky", "flags": ["jerky"], "...": "..."}}}
```
- `episodes` is keyed by the **exported** `episode_index`. `source_episode_index` is mandatory because `--exclude-label fail` renumbers episodes.
- The `dataset` fingerprint (count, frames, lengths) must match `info.json`. A mismatch means the sidecar is stale, for example after a lerobot edit tool (G5), so warn or refuse.
- Keep per-frame `next.success` / `next.reward` as real features. They are standard LeRobot keys, get stats for free, and survive lerobot edit tools.
- Considered and rejected:
  - (a) Extra `rrc/*` columns in `meta/episodes/*.parquet`: loads fine and shows up in `ds.meta.episodes`, but it needs a post-`finalize` parquet rewrite, is dropped by lerobot tools, and can collide with future lerobot columns. Maybe mirror into it later, read-only.
  - (b) Encoding tags in the `task` string: it pollutes `tasks.parquet`/`task_index` and the language conditioning of VLA policies (SmolVLA etc.).

## Open uncertainties
- Mac install size and time for `lerobot`, and whether the `mujoco.Renderer` default works on a MacBook Air: not tested (no Mac). The founder should run the spike once on the Mac.
- CPU-torch CI install: not tested (pytorch index blocked here).
- lerobot upstream may fix G1 or change APIs in 0.5. Re-verify when bumping the pin.

## Changes applied (Researcher, after QA review 2026-09-24)
- R1: Tags now follow a session-first model: `episodes.jsonl` is the source of truth, and export writes a snapshot to `meta/rrc_tags.json` (schema_version, fingerprint, exported-index keys, `source_episode_index`, separate `sim_success`/`label`). Updated in Rec. 4, §4 and the spike. The spike's old `rrc_tags.jsonl` is gone.
- R2: SO-101 MJCF + 13 STL meshes + Apache-2.0 LICENSE + SOURCE.md are vendored at `src/robot_report_card/sim/assets/so101/` (16 MB). The spike loads from there with no `robot_descriptions`, and it re-ran offline with `HF_HUB_OFFLINE=1`: 4 episodes / 360 frames reloaded and the fingerprint asserted. Note: the phase 1 plan says `src/robot_report_card/assets/so101/`. I used the path the coordinator gave (`sim/assets/`), so the plan should be updated to match.
- R3: Added G7 (missing or half-written root falls through to the Hub and leaves an empty `meta/`) and G8 (`timestep = 1/(30*17)`, asserted in the spike).
- R4: Success rates re-measured with one RNG per seed, seeds 0–99, corrected timestep: smooth 77% (seeds 0–19: 70%), jerky 8%, stall 0%. The seeding method is stated in §3. Finding 6: the episodes-table column count is corrected (7 + 10·F, 109 here).

## QA Review (2026-09-24)

**Verdict: APPROVED WITH CHANGES.** The main factual claims hold up against the installed lerobot 0.4.4 source and against runs I did myself. The spike round-trips. The changes below are about internal consistency and about testing offline, not about correctness.

**Verified (by QA, in `/home/claude/.venvs/rrc`, Python 3.11.15):**
- Signatures of `create` / `add_frame` / `save_episode` / `finalize` / `__init__` match §2 exactly (`inspect.signature`). `CODEBASE_VERSION = "v3.0"`. `create` does `mkdir(exist_ok=False)` (G2).
- Spike: `reloaded: 4 episodes 360 frames`, 5.4 s wall. On-disk layout, `info.json` keys, fixed_size_list/scalar parquet columns, stats keys and HF schema footer all match §1.
- **G1 reproduced.** Same spike in an overlay venv with numpy 2.4.6 gives `TypeError: only 0-dimensional arrays can be converted to Python scalars` (raised inside `datasets` `float(value)` for the shape-`(1,)` features). With no user `(1,)` features, numpy 2.4 works. The pin is needed because we keep `next.reward` / `next.success`.
- Without `finalize()` (writer still open in the same process), reload fails. It then **falls back to a Hub request** (`pull_from_repo`), which with `HF_HUB_OFFLINE=1` raises `OfflineModeIsEnabled`. The claim holds, and see finding 4.
- Env spec: joints and actuators are `"1".."6"`, kp=17.8, dt=0.002, site `gripper`, 70 eps/s. All confirmed. The `smooth` success rate depends on how seeds are drawn: 86/100 with one shared RNG (the spike's way), but **14/20 on seeds 0–19 and 77/100 with one RNG per seed** (how a `--seed` CLI works). So "85%" overstates it. Quote about 75%.
- Tags sidecar: extra files in `meta/` don't affect loading. `push_to_hub` ignores only `images/` (and `videos/` when that's turned off).

**Findings:**
1. **The spike contradicts rec. 4.** The brief says `meta/rrc_tags.json` (a dict keyed by `episode_index`, with a `dataset` fingerprint). The spike writes `meta/rrc_tags.jsonl` (a list, no fingerprint). The phase 1 plan uses a third name (`meta/rrc_episodes.jsonl`). The team needs one name and one schema (see the plan's QA review).
2. **The tag schema merges ground truth and human labels.** The single `success` bool plus `tagged_by` loses the sim result as soon as a human overrides it. The plan keeps `sim_success` and `label` (success|fail|unlabeled) separate, which is what phase 2 needs. Use the plan's fields.
3. **Asset fetch vs. "no network in tests".** `import robot_descriptions.so_arm101_mj_description` git-clones 295 MB **at import time** (GitPython, so it needs a `git` binary and GitHub access). Caching `~/.cache/robot_descriptions` doesn't meet the plan's DoD 6 on a fresh machine. QA checked that `Simulation/SO101/so101_new_calib.xml` + `assets/*.stl` (16 MB, `.part` files not needed) loads fine as a standalone copy. **Vendor it with the Apache-2.0 LICENSE** and drop `robot_descriptions` as a runtime dependency.
4. **A load on a missing or half-written root goes to the Hub.** `LeRobotDataset(repo_id, root=...)` catches FileNotFoundError and calls `pull_from_repo`. Tests must set `HF_HUB_OFFLINE=1`, or a bad path turns into a network call or a hang instead of a clean failure.
5. **fps and the physics grid don't line up.** 17 substeps × 0.002 s = 0.034 s per frame, not 1/30 s, so the sim runs about 2% slow compared with the stamped timestamps. Fix it by setting `m.opt.timestep = 1/(fps*17)` (or any substep count that divides evenly), and add an assertion.
6. Minor: the episodes table has **109** columns for the brief's own 5-feature set, not 99 (the count is 7 + 10 × number of features). Python 3.10 resolves to numpy ≤2.2 (numpy 2.3 needs ≥3.11). That's allowed, but untested.

**Required changes:**
- R1: Make the spike write the same sidecar name and schema the team adopts (`meta/rrc_tags.json`, keyed by episode_index, `schema_version`, fingerprint, with `sim_success` and `label` separate), or mark the spike's sidecar as superseded.
- R2: Replace "cache for now" with "vendor `Simulation/SO101` (xml + STL + LICENSE)".
- R3: Add findings 4 and 5 to the gotchas (G7 offline load, G8 timestep).
- R4: Change the `smooth` figure in §3 to about 75% (one RNG per seed) and say how seeds were drawn.

### QA Re-review (2026-09-24)
**Verdict: APPROVED.** R1–R4 are applied. The vendored files are byte-identical to upstream at commit `63eede5` (xml, scene, 13 STL; LICENSE and SOURCE.md included). I re-ran the spike with `HF_HUB_OFFLINE=1`, a dead proxy and `robot_descriptions` blocked from importing: 4 episodes / 360 frames reloaded, `meta/rrc_tags.json` fingerprint asserted, and `sim_success` / `label` kept separate. `--measure` reproduces smooth 0.77 (0.70 on seeds 0–19), jerky 0.08, stall 0.00. Non-blocking: the path-mismatch note in "Changes applied" R2 is out of date, because the plan now uses `sim/assets/so101/`.

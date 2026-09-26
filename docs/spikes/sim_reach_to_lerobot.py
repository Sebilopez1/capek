"""Research spike (not production code): SO-101 MuJoCo reach task -> LeRobot v3.0 dataset + rrc_tags.json -> reload.
Verified 2026-09-24 with lerobot==0.4.4, numpy==2.3.5, mujoco==3.14.0, Python 3.11. No robot_descriptions / network needed.
Run:  HF_HUB_OFFLINE=1 /home/claude/.venvs/rrc/bin/python docs/spikes/sim_reach_to_lerobot.py [out_dir]   (out_dir must NOT exist)
      ... sim_reach_to_lerobot.py --measure    # success rates only, 100 episodes per style, no export
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import mujoco, numpy as np

ASSET = Path(__file__).resolve().parents[2] / "src/robot_report_card/sim/assets/so101/so101_new_calib.xml"
FPS, SUBSTEPS, T = 30, 17, 90
m = mujoco.MjModel.from_xml_path(str(ASSET)); x = mujoco.MjData(m)
m.opt.timestep = 1.0 / (FPS * SUBSTEPS)                      # G8: physics grid == timestamp grid
assert abs(SUBSTEPS * m.opt.timestep - 1.0 / FPS) < 1e-12
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper"); lo, hi = m.jnt_range.T
NAMES = [f"{j}.pos" for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]

def run_episode(seed, style):
    rng = np.random.default_rng(seed)                        # one RNG per episode -> reproducible alone
    mujoco.mj_resetData(m, x); qt = rng.uniform(lo * .6, hi * .6)
    x.qpos[:] = qt; mujoco.mj_forward(m, x); tgt = x.site_xpos[site].copy(); mujoco.mj_resetData(m, x)
    frames, prev = [], None
    for t in range(T):
        a = qt * min(1, (t + 1) / 60)
        if style == "jerky": a = a + rng.normal(0, .25, 6)
        if style == "stall" and t > 30: a = prev
        prev = a; state = x.qpos.astype(np.float32).copy(); x.ctrl[:] = a
        for _ in range(SUBSTEPS): mujoco.mj_step(m, x)
        err = float(np.linalg.norm(x.site_xpos[site] - tgt))
        frames.append((state, tgt.astype(np.float32), a.astype(np.float32), -err, err < 0.02))
    return frames, err

if "--measure" in sys.argv:
    for style in ["smooth", "jerky", "stall"]:
        errs = np.array([run_episode(s, style)[1] for s in range(100)])
        print(f"{style}: success {np.mean(errs < 0.02):.2f} over seeds 0-99 (seeds 0-19: {np.mean(errs[:20] < 0.02):.2f}), median err {np.median(errs)*100:.1f} cm")
    sys.exit()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/spike_reach"
features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
            "observation.environment_state": {"dtype": "float32", "shape": (3,), "names": ["target_x", "target_y", "target_z"]},
            "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.success": {"dtype": "bool", "shape": (1,), "names": None}}
ds = LeRobotDataset.create(repo_id="rrc/sim_reach_demo", fps=FPS, features=features, root=OUT, robot_type="so101_sim", use_videos=False)
session = []                                                 # stands in for the session's episodes.jsonl rows
for ep, style in enumerate(["smooth", "jerky", "smooth", "stall"]):
    frames, err = run_episode(seed=ep, style=style)
    for s, tg, a, r, ok in frames:
        ds.add_frame({"observation.state": s, "observation.environment_state": tg, "action": a,
                      "next.reward": np.array([r], np.float32), "next.success": np.array([ok]),
                      "task": "Move the gripper tip to the target point."})
    ds.save_episode()
    session.append({"episode_index": ep, "env_id": "so101_reach", "policy_name": "scripted", "policy_params": {"style": style},
                    "seed": ep, "fps": FPS, "num_frames": T, "duration_s": T / FPS, "sim_success": err < 0.02,
                    "final_error_m": round(err, 4), "termination_reason": "max_steps", "label": "unlabeled",
                    "notes": "", "flags": [], "recorded_at": datetime.now(timezone.utc).isoformat(), "rrc_version": "0.0.0-spike"})
ds.finalize()
session[1].update(label="fail", notes="looked shaky", flags=["jerky"])   # a human tag, kept separate from sim_success
info = json.load(open(f"{OUT}/meta/info.json"))
tags = {"schema_version": 1, "tool": "robot-report-card", "rrc_version": "0.0.0-spike",
        "dataset": {"codebase_version": info["codebase_version"], "total_episodes": info["total_episodes"],
                    "total_frames": info["total_frames"], "episode_lengths": [e["num_frames"] for e in session]},
        "episodes": {str(i): {"source_episode_index": e["episode_index"], **{k: v for k, v in e.items() if k != "episode_index"}}
                     for i, e in enumerate(session)}}
Path(f"{OUT}/meta/rrc_tags.json").write_text(json.dumps(tags, indent=2))
r = LeRobotDataset("rrc/sim_reach_demo", root=OUT)
t2 = json.load(open(f"{OUT}/meta/rrc_tags.json"))
assert t2["dataset"]["total_frames"] == r.num_frames and len(t2["episodes"]) == r.num_episodes
print("reloaded:", r.num_episodes, "episodes", r.num_frames, "frames; sample keys", sorted(r[100].keys()))
print({i: (e["sim_success"], e["label"], e["final_error_m"]) for i, e in t2["episodes"].items()})

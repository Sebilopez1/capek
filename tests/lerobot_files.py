"""Write small LeRobot-layout datasets with pyarrow (v3.0 multi-episode files, v2.x per-episode files).

Test scaffolding for the reader and scorer; not the QA benchmark.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def _vector(pa: Any, x: np.ndarray, list_type: str) -> Any:
    x = np.asarray(x, dtype=np.float32)
    flat = pa.array(x.reshape(-1), type=pa.float32())
    if list_type == "fixed":
        return pa.FixedSizeListArray.from_arrays(flat, x.shape[1])
    offsets = pa.array(np.arange(0, x.size + 1, x.shape[1], dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat)


def write_dataset(
    root: Path,
    episodes: list[dict[str, np.ndarray]],
    version: str = "v3.0",
    fps: int = 30,
    names: list[str] | None = None,
    tags: dict[str, Any] | None = None,
    files: int = 2,
    shuffle: bool = True,
    info_overrides: dict[str, Any] | None = None,
) -> Path:
    """``episodes``: dicts with ``observation.state`` (T, D), ``action`` (T, A), optional ``next.success`` (T,)
    and optional ``timestamp`` (T,). v3 rows are shuffled across ``files`` files to exercise the sort."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = names or JOINTS
    root = Path(root)
    (root / "meta").mkdir(parents=True)
    tables = []
    offset = 0
    for i, ep in enumerate(episodes):
        s, a = ep["observation.state"], ep["action"]
        n = len(s)
        cols = {
            "observation.state": _vector(pa, s, "fixed" if version.startswith("v3") else "list"),
            "action": _vector(pa, a, "fixed" if version.startswith("v3") else "list"),
            "timestamp": pa.array(ep.get("timestamp", np.arange(n) / fps).astype(np.float32)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, i, dtype=np.int64)),
            "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
        }
        if "next.success" in ep:
            cols["next.success"] = pa.array(np.asarray(ep["next.success"], dtype=bool))
        tables.append(pa.table(cols))
        offset += n
    features = {
        "observation.state": {"dtype": "float32", "shape": [episodes[0]["observation.state"].shape[1]], "names": names},
        "action": {"dtype": "float32", "shape": [episodes[0]["action"].shape[1]], "names": names},
    }
    info = {
        "codebase_version": version,
        "robot_type": "test",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "features": features,
        **(info_overrides or {}),
    }
    if version.startswith("v3"):
        table = pa.concat_tables(tables)
        order = np.random.default_rng(0).permutation(table.num_rows) if shuffle else np.arange(table.num_rows)
        table = table.take(pa.array(order))
        for k, part in enumerate(np.array_split(np.arange(table.num_rows), files)):
            path = root / "data" / f"chunk-{k:03d}" / f"file-{k:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.take(pa.array(part)), path)
    else:
        for i, t in enumerate(tables):
            path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, path)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    if tags is not None:
        (root / "meta" / "rrc_tags.json").write_text(json.dumps(tags))
    return root


def v3_to_v2(src: Path, dst: Path, version: str = "v2.1") -> Path:
    """Copy a v3.0 dataset (e.g. an `rrc export`) into the per-episode v2.x layout."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pa.concat_tables(pq.read_table(f) for f in sorted(Path(src).glob("data/*/file-*.parquet")))
    (Path(dst) / "meta").mkdir(parents=True)
    for e in sorted(set(table.column("episode_index").to_pylist())):
        part = table.filter(pc.equal(table.column("episode_index"), e))
        path = Path(dst) / "data" / f"chunk-{e // 1000:03d}" / f"episode_{e:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(part, path)
    info = json.loads((Path(src) / "meta" / "info.json").read_text())
    info["codebase_version"] = version
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    (Path(dst) / "meta" / "info.json").write_text(json.dumps(info))
    if (Path(src) / "meta" / "rrc_tags.json").is_file():
        shutil.copy(Path(src) / "meta" / "rrc_tags.json", Path(dst) / "meta" / "rrc_tags.json")
    return Path(dst)


def smooth_reach(rng: np.random.Generator, n: int = 90, noise: float = 0.0) -> dict[str, np.ndarray]:
    """A clean-ish reach: ramp to a random goal then hold, action leads state by one frame."""
    goal = rng.uniform(-1, 1, 6)
    ramp = np.clip(np.arange(1, n + 1) / rng.uniform(30, 75), 0, 1)[:, None]
    action = ramp * goal + rng.normal(0, noise, (n, 6)) if noise else ramp * goal
    state = np.vstack([np.zeros((1, 6)), action[:-1]]) * 0.98
    return {"observation.state": state, "action": action, "next.success": np.r_[np.zeros(n - 1), 1].astype(bool)}

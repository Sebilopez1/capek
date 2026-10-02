"""Real-like and corrupt fixtures derived from a capek-exported LeRobot v3.0 dataset (pyarrow + numpy only).

``realify SRC --out DIR`` writes (each a complete dataset root) plus ``DIR/manifest.json``:

- ``units``      v3.0; arm joints rad -> degrees, gripper -> 0..100 (affine over the data's gripper range),
                 joints renamed to the older ``main_<joint>`` style; float32 as on disk. (plan variant a)
- ``truncated``  v3.0; 20% of episodes cut to a random length in [30, T-1]. (variant b)
- ``v21``/``v20`` the same data in the one-parquet-per-episode v2.1 / v2.0 layouts (variant c); the v2.0 copy
                 nests joint names as ``{"motors": [...]}`` like v2.0-era SO-100 datasets.
- ``video_declared`` v3.0; ``info.json`` declares a camera ``video`` feature (and ``video_path``) but ships no
                 ``videos/``, like a Hub download of ``meta/`` + ``data/`` only.
- ``corrupt_nan`` / ``corrupt_frozen`` / ``corrupt_dropped``  12-episode v3.0 copies with one defect each.

The v3.0 rewrites update ``info.json`` totals, the episodes table's length/from/to columns and the
``capek_tags.json`` fingerprint; per-episode ``stats/*`` columns in ``meta/episodes`` are left as in the source
(stale for truncated/units copies; nothing in capek reads them). ``meta/stats.json`` is transformed for ``units``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from capek.legacy import TAGS_FILES, find_file

STATE, ACTION = "observation.state", "action"
ARM_JOINTS, GRIPPER = slice(0, 5), 5
OLD_NAMES = [
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper",
]
TRUNC_FRAC, TRUNC_MIN = 0.2, 30
CORRUPT_EPISODES = 12
CORRUPT = {  # fixture -> (episode, expected hard-flag prefix in the capek score report)
    "corrupt_nan": (3, "non-finite"),
    "corrupt_frozen": (5, "frozen joint"),
    "corrupt_dropped": (7, "timestamp gap"),
}


def _pa() -> tuple[Any, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


class Source:
    def __init__(self, root: Path) -> None:
        pa, pq = _pa()
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text())
        if not str(self.info.get("codebase_version", "")).startswith("v3"):
            raise ValueError(f"realify needs a v3.x source dataset, got {self.info.get('codebase_version')}")
        files = sorted(root.glob("data/*/file-*.parquet"))
        self.schema = pq.read_schema(files[0])
        table = pa.concat_tables([pq.read_table(f) for f in files])
        order = np.lexsort((table.column("frame_index").to_numpy(), table.column("episode_index").to_numpy()))
        self.table = table.take(pa.array(order))
        ep = self.table.column("episode_index").to_numpy()
        self.episode_ids = [int(e) for e in np.unique(ep)]
        self.slices = {e: np.nonzero(ep == e)[0] for e in self.episode_ids}
        self.episodes_meta = pa.concat_tables(
            [pq.read_table(f) for f in sorted(root.glob("meta/episodes/*/*.parquet"))]
        )
        tags = find_file(root / "meta", TAGS_FILES)
        self.tags = json.loads(tags.read_text()) if tags.is_file() else None

    def column(self, name: str) -> np.ndarray:
        col = self.table.column(name).combine_chunks()
        if hasattr(col.type, "list_size"):
            return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), col.type.list_size)
        return col.to_numpy(zero_copy_only=False)


def _vector_array(values: np.ndarray) -> Any:
    pa, _ = _pa()
    values = np.ascontiguousarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def _frames(src: Source, rows: np.ndarray, replace: dict[str, np.ndarray] | None = None) -> Any:
    """Table of the given source rows (in order), with replaced columns; ``index`` renumbered 0..n-1."""
    pa, _ = _pa()
    t = src.table.take(pa.array(rows))
    for name, values in (replace or {}).items():
        arr = _vector_array(values) if values.ndim == 2 else pa.array(values, type=t.schema.field(name).type)
        t = t.set_column(t.schema.get_field_index(name), name, arr)
    t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(np.arange(len(t), dtype=np.int64)))
    return t.cast(src.schema)  # keep the source schema (incl. the HF metadata footer)


def _copy_meta(src: Source, dst: Path) -> None:
    (dst / "meta").mkdir(parents=True)
    for name in ("tasks.parquet", "stats.json"):
        if (src.root / "meta" / name).is_file():
            shutil.copy2(src.root / "meta" / name, dst / "meta" / name)


def _write_v3(src: Source, dst: Path, table: Any, lengths: dict[int, int], info_updates: dict | None = None) -> None:
    """Write a v3.0 root holding ``table`` (episodes in ``lengths`` order) with consistent meta."""
    pa, pq = _pa()
    _copy_meta(src, dst)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, dst / "data" / "chunk-000" / "file-000.parquet")
    ids = list(lengths)
    meta = src.episodes_meta
    rows = [int(np.nonzero(meta.column("episode_index").to_numpy() == e)[0][0]) for e in ids]
    meta = meta.take(pa.array(rows))
    ends = np.cumsum([lengths[e] for e in ids])
    starts = ends - np.array([lengths[e] for e in ids])
    upd = {
        "length": [lengths[e] for e in ids],
        "dataset_from_index": starts,
        "dataset_to_index": ends,
        "data/chunk_index": np.zeros(len(ids)),
        "data/file_index": np.zeros(len(ids)),
        "meta/episodes/chunk_index": np.zeros(len(ids)),
        "meta/episodes/file_index": np.zeros(len(ids)),
    }
    for name, values in upd.items():
        i = meta.schema.get_field_index(name)
        meta = meta.set_column(i, name, pa.array(np.asarray(values, dtype=np.int64), type=meta.schema.field(i).type))
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(meta, dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = json.loads(json.dumps(src.info))
    info.update(total_episodes=len(ids), total_frames=int(ends[-1]), splits={"train": f"0:{len(ids)}"})
    info.update(info_updates or {})
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    _write_tags(src, dst, info, lengths)


def _write_tags(src: Source, dst: Path, info: dict, lengths: dict[int, int]) -> None:
    if src.tags is None:
        return
    tags = json.loads(json.dumps(src.tags))
    eps = {}
    for new, e in enumerate(lengths):
        row = tags["episodes"][str(e)]
        row.update(num_frames=lengths[e], duration_s=lengths[e] / info["fps"])
        eps[str(new)] = row
    tags["episodes"] = eps
    tags["dataset"].update(
        total_episodes=len(lengths),
        total_frames=sum(lengths.values()),
        episode_lengths=list(lengths.values()),
        codebase_version=info["codebase_version"],
    )
    (dst / "meta" / "capek_tags.json").write_text(json.dumps(tags, indent=2))


def _all_rows(src: Source, ids: list[int]) -> np.ndarray:
    return np.concatenate([src.slices[e] for e in ids])


# ---- variants -------------------------------------------------------------------------------------------------
def units(src: Source, dst: Path) -> dict[str, Any]:
    """(a) degrees + 0..100 gripper + renamed joints; returns the affine map per joint."""
    s, a = src.column(STATE).astype(np.float64), src.column(ACTION).astype(np.float64)
    g_lo = float(min(s[:, GRIPPER].min(), a[:, GRIPPER].min()))
    g_hi = float(max(s[:, GRIPPER].max(), a[:, GRIPPER].max()))
    scale = np.array([180 / np.pi] * 5 + [100 / (g_hi - g_lo)])
    offset = np.array([0.0] * 5 + [-g_lo * 100 / (g_hi - g_lo)])
    ids = src.episode_ids
    rows = _all_rows(src, ids)
    table = _frames(src, rows, {STATE: s[rows] * scale + offset, ACTION: a[rows] * scale + offset})
    feats = json.loads(json.dumps(src.info["features"]))
    for key in (STATE, ACTION):
        feats[key]["names"] = list(OLD_NAMES)
    _write_v3(
        src, dst, table, {e: len(src.slices[e]) for e in ids}, {"features": feats, "robot_type": "so101_follower"}
    )
    stats_path = dst / "meta" / "stats.json"
    if stats_path.is_file():  # keep dataset-level stats in the new units
        stats = json.loads(stats_path.read_text())
        for key in (STATE, ACTION):
            for stat, vals in stats.get(key, {}).items():
                v = np.asarray(vals, dtype=np.float64)
                if stat == "std":
                    stats[key][stat] = (v * np.abs(scale)).tolist()
                elif stat != "count":
                    stats[key][stat] = (v * scale + offset).tolist()
        stats_path.write_text(json.dumps(stats, indent=4))
    return {"scale": scale.tolist(), "offset": offset.tolist(), "names": OLD_NAMES}


def truncated(src: Source, dst: Path, seed: int = 0) -> dict[str, Any]:
    """(b) 20% of episodes cut to a random length in [TRUNC_MIN, T-1]."""
    rng = np.random.default_rng(seed)
    ids = src.episode_ids
    chosen = sorted(int(e) for e in rng.choice(ids, max(1, round(TRUNC_FRAC * len(ids))), replace=False))
    lengths = {}
    rows = []
    for e in ids:
        r = src.slices[e]
        n = int(rng.integers(TRUNC_MIN, len(r))) if e in chosen and len(r) > TRUNC_MIN else len(r)
        rows.append(r[:n])
        lengths[e] = n
    _write_v3(src, dst, _frames(src, np.concatenate(rows)), lengths)
    return {"truncated_episodes": chosen, "lengths": {str(e): lengths[e] for e in chosen}}


def v2(src: Source, dst: Path, version: str) -> dict[str, Any]:
    """(c) one parquet per episode, v2.1 (episodes_stats.jsonl) or v2.0 (stats.json) meta."""
    pa, pq = _pa()
    (dst / "meta").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    task_table = (
        pq.read_table(src.root / "meta" / "tasks.parquet").to_pandas()
        if (src.root / "meta" / "tasks.parquet").is_file()
        else None
    )
    tasks = {int(r.task_index): str(idx) for idx, r in task_table.iterrows()} if task_table is not None else {0: ""}
    ep_lines, stat_lines = [], []
    total = 0
    for e in src.episode_ids:
        t = _frames(src, src.slices[e])
        t = t.set_column(
            t.schema.get_field_index("index"), "index", pa.array(np.arange(total, total + len(t), dtype=np.int64))
        )
        pq.write_table(t, dst / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        task_ids = sorted({int(x) for x in t.column("task_index").to_numpy()})
        ep_lines.append({"episode_index": e, "tasks": [tasks.get(i, "") for i in task_ids], "length": len(t)})
        stats = {}
        for key in (STATE, ACTION):
            col = t.column(key).combine_chunks()
            v = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float64)
            stats[key] = {
                "min": v.min(0).tolist(),
                "max": v.max(0).tolist(),
                "mean": v.mean(0).tolist(),
                "std": v.std(0).tolist(),
                "count": [len(v)],
            }
        stat_lines.append({"episode_index": e, "stats": stats})
        total += len(t)
    info = {
        "codebase_version": version,
        "robot_type": src.info.get("robot_type"),
        "total_episodes": len(src.episode_ids),
        "total_frames": total,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": src.info["fps"],
        "splits": {"train": f"0:{len(src.episode_ids)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": _v2_features(src.info["features"], nested_names=version == "v2.0"),
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (dst / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_lines))
    (dst / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in tasks.items())
    )
    if version == "v2.1":
        (dst / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stat_lines))
    elif (src.root / "meta" / "stats.json").is_file():
        stats = json.loads((src.root / "meta" / "stats.json").read_text())
        stats = {k: {s: v for s, v in d.items() if s in ("min", "max", "mean", "std")} for k, d in stats.items()}
        (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    _write_tags(src, dst, info, {e: len(src.slices[e]) for e in src.episode_ids})
    return {"codebase_version": version, "nested_names": version == "v2.0"}


def _v2_features(features: dict[str, Any], nested_names: bool) -> dict[str, Any]:
    feats = json.loads(json.dumps(features))
    if nested_names:
        for key in (STATE, ACTION):
            if isinstance(feats.get(key, {}).get("names"), list):
                feats[key]["names"] = {"motors": feats[key]["names"]}
    return feats


VIDEO_KEY = "observation.images.front"


def video_declared(src: Source, dst: Path) -> dict[str, Any]:
    """v3.0 copy whose info.json declares a camera video feature, with no videos/ directory shipped."""
    feats = json.loads(json.dumps(src.info["features"]))
    feats[VIDEO_KEY] = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.fps": src.info["fps"], "video.codec": "av1", "video.pix_fmt": "yuv420p", "has_audio": False},
    }
    ids = src.episode_ids
    _write_v3(
        src,
        dst,
        _frames(src, _all_rows(src, ids)),
        {e: len(src.slices[e]) for e in ids},
        {"features": feats, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"},
    )
    return {"video_feature": VIDEO_KEY, "videos_dir_present": (dst / "videos").exists()}


def corrupt(src: Source, dst: Path, kind: str) -> dict[str, Any]:
    ids = src.episode_ids[:CORRUPT_EPISODES]
    bad, expected = CORRUPT[kind]
    s = src.column(STATE).astype(np.float32)
    ts = src.column("timestamp").astype(np.float32)
    fi = src.column("frame_index").astype(np.int64)
    rows, lengths = [], {}
    for e in ids:
        r = src.slices[e]
        if kind == "corrupt_dropped" and e == ids[bad]:
            r = np.concatenate(
                [r[:40], r[45:]]
            )  # 5 frames lost; the recorder renumbers frames, timestamps keep the gap
        rows.append(r)
        lengths[e] = len(r)
    rows = np.concatenate(rows)
    state, stamps, frames = s[rows].copy(), ts[rows].copy(), fi[rows].copy()
    ep = src.column("episode_index")[rows]
    sel = np.nonzero(ep == ids[bad])[0]
    if kind == "corrupt_nan":
        state[sel[40], 2] = np.nan
    elif kind == "corrupt_frozen":
        state[sel, 1] = state[sel[0], 1]
    elif kind == "corrupt_dropped":
        frames[sel] = np.arange(len(sel))
    _write_v3(src, dst, _frames(src, rows, {STATE: state, "timestamp": stamps, "frame_index": frames}), lengths)
    return {"episode": int(ids[bad]), "expected_hard_flag": expected}


def realify(src_root: Path | str, out: Path | str, seed: int = 0) -> dict[str, Any]:
    src_root, out = Path(src_root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new --out")
    src = Source(src_root)
    manifest: dict[str, Any] = {
        "source": str(src_root.resolve()),
        "source_codebase_version": src.info["codebase_version"],
    }
    manifest["units"] = units(src, out / "units")
    manifest["truncated"] = truncated(src, out / "truncated", seed)
    manifest["v21"] = v2(src, out / "v21", "v2.1")
    manifest["v20"] = v2(src, out / "v20", "v2.0")
    manifest["video_declared"] = video_declared(src, out / "video_declared")
    for kind in CORRUPT:
        manifest[kind] = corrupt(src, out / kind, kind)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

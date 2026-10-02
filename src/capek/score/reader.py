"""Read a LeRobot dataset from disk with pyarrow + numpy only (no lerobot, no torch).

Supported layouts (dispatch on ``meta/info.json`` ``codebase_version``):

- ``v3.x``: ``data/chunk-*/file-*.parquet``, many episodes per file.
- ``v2.0`` / ``v2.1``: ``data/chunk-*/episode_*.parquet``, one episode per file (same per-frame columns).

Rows are sorted by (``episode_index``, ``frame_index``) and split into per-episode float64 arrays. The optional
``meta/capek_tags.json`` sidecar (written by ``capek export``) is returned for outcome labels; the dataset is never
modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from capek.hints import install_hint
from capek.legacy import TAGS_FILES, find_file

V3_GLOB = "data/*/file-*.parquet"
V2_GLOB = "data/*/episode_*.parquet"
SUCCESS_KEY = "next.success"
INDEX_COLUMNS = ("episode_index", "frame_index", "timestamp")


class ReaderError(Exception):
    """A user-facing problem reading a dataset (missing files, unsupported version, missing keys...)."""


@dataclass
class EpisodeData:
    index: int  # episode_index as stored in the dataset
    state: np.ndarray  # (T, D) float64
    action: np.ndarray  # (T, A) float64
    timestamp: np.ndarray  # (T,) float64
    frame_index: np.ndarray  # (T,) int64
    success: np.ndarray | None  # (T,) bool, when the dataset has next.success

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


@dataclass
class Dataset:
    root: Path
    info: dict[str, Any]
    codebase_version: str
    fps: float
    state_key: str
    action_key: str
    state_names: list[str] | None
    action_names: list[str] | None
    episodes: list[EpisodeData]
    tags: dict[str, Any] | None = None  # capek_tags.json, only when its fingerprint matches info.json
    notes: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return sum(e.length for e in self.episodes)


def layout_for(version: str) -> str:
    """Return the data-file glob for a ``codebase_version``, or raise ReaderError."""
    if version.startswith("v3."):
        return V3_GLOB
    if version in ("v2.0", "v2.1"):
        return V2_GLOB
    raise ReaderError(f"unsupported codebase_version {version!r}: capek reads LeRobot v3.x, v2.1 and v2.0 datasets")


def _import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ReaderError(f"`capek score` needs the score extra: {install_hint('score')} (pyarrow)") from e
    return pq


def _to_2d(column: Any, name: str) -> np.ndarray:
    """A pyarrow column of fixed-size lists, lists or scalars -> (n, d) float64."""
    import pyarrow as pa

    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    n = len(arr)
    if pa.types.is_fixed_size_list(arr.type) or pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        if pa.types.is_fixed_size_list(arr.type):
            d = arr.type.list_size
        else:
            lengths = np.asarray(arr.value_lengths().fill_null(0).to_numpy(zero_copy_only=False))
            if n and not np.all(lengths == lengths[0]):
                raise ReaderError(f"column {name!r} has rows of different lengths")
            d = int(lengths[0]) if n else 0
        values = arr.flatten().to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float64).reshape(n, d)
    if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type) or pa.types.is_boolean(arr.type):
        return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(n, 1)
    raise ReaderError(f"column {name!r} has unsupported type {arr.type}")


def _feature_names(info: dict[str, Any], key: str) -> list[str] | None:
    names = info.get("features", {}).get(key, {}).get("names")
    if isinstance(names, dict):  # some datasets nest names, e.g. {"motors": [...]}
        names = next(iter(names.values()), None)
    return [str(n) for n in names] if isinstance(names, list) else None


def _vector_keys(info: dict[str, Any]) -> list[str]:
    feats = info.get("features", {})
    return sorted(k for k, v in feats.items() if v.get("dtype") in ("float32", "float64") and v.get("shape"))


def _read_tags(root: Path, info: dict[str, Any], notes: list[str]) -> dict[str, Any] | None:
    path = find_file(root / "meta", TAGS_FILES)
    if not path.is_file():
        return None
    try:
        tags = json.loads(path.read_text())
        fp = tags["dataset"]
        stale = [k for k in ("total_episodes", "total_frames") if fp.get(k) != info.get(k)]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        notes.append(f"meta/{path.name} is unreadable ({e}); labels ignored")
        return None
    if stale:
        notes.append(f"meta/{path.name} is stale ({', '.join(stale)} differ from info.json); labels ignored")
        return None
    return tags


def read_dataset(root: Path | str, state_key: str = "observation.state", action_key: str = "action") -> Dataset:
    """Load every episode's state/action/timestamps (+ ``next.success`` if present) from ``root``."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ReaderError(f"{root} is not a LeRobot dataset (no meta/info.json)")
    try:
        info = json.loads(info_path.read_text())
    except ValueError as e:
        raise ReaderError(f"{info_path} is not valid JSON: {e}") from e
    version = str(info.get("codebase_version", "?"))
    pattern = layout_for(version)
    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ReaderError(f"{info_path} has no valid fps")
    files = sorted(root.glob(pattern))
    if not files:
        raise ReaderError(f"no data files matching {pattern} under {root} (codebase_version {version})")

    pq = _import_pyarrow()
    notes: list[str] = []
    try:
        schemas = [pq.read_schema(f) for f in files]
    except Exception as e:  # pyarrow raises several types for unreadable footers
        raise ReaderError(f"could not read parquet metadata under {root}: {e}") from e
    schema_names = set(schemas[0].names)
    for key, flag in ((state_key, "--state-key"), (action_key, "--action-key")):
        if key not in schema_names:
            options = ", ".join(k for k in _vector_keys(info) if k in schema_names) or "none"
            raise ReaderError(f"dataset has no {key!r} column (pass {flag}; vector features: {options})")
    missing = [c for c in INDEX_COLUMNS if c not in schema_names]
    if missing:
        raise ReaderError(f"dataset is missing required column(s): {', '.join(missing)}")
    columns = list(dict.fromkeys([state_key, action_key, *INDEX_COLUMNS]))
    has_success = all(SUCCESS_KEY in sch.names for sch in schemas)
    if has_success:
        columns.append(SUCCESS_KEY)
    elif any(SUCCESS_KEY in sch.names for sch in schemas):
        notes.append(f"{SUCCESS_KEY} is missing from some data files; sim outcome ignored")
    for f, sch in zip(files, schemas, strict=True):
        if any(c not in sch.names or sch.field(c).type != schemas[0].field(c).type for c in columns):
            raise ReaderError(
                f"episode files have different columns or types (first differing file: {f.relative_to(root)})"
            )

    import pyarrow as pa

    try:
        table = pa.concat_tables([pq.read_table(f, columns=columns) for f in files])
    except (pa.ArrowException, OSError) as e:
        raise ReaderError(f"could not read parquet data under {root}: {e}") from e
    state = _to_2d(table.column(state_key), state_key)
    action = _to_2d(table.column(action_key), action_key)
    ep_idx = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    fr_idx = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
    ts = np.asarray(table.column("timestamp").to_numpy(zero_copy_only=False), dtype=np.float64)
    success = _to_2d(table.column(SUCCESS_KEY), SUCCESS_KEY)[:, 0] > 0.5 if has_success else None

    for key, arr in ((state_key, state), (action_key, action)):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape and int(np.prod(shape)) != arr.shape[1]:
            raise ReaderError(f"{key!r} has {arr.shape[1]} dims in the data but shape {shape} in info.json")

    order = np.lexsort((fr_idx, ep_idx))
    state, action, ep_idx, fr_idx, ts = state[order], action[order], ep_idx[order], fr_idx[order], ts[order]
    if success is not None:
        success = success[order]
    uniq, starts = np.unique(ep_idx, return_index=True)
    bounds = [*starts[1:], len(ep_idx)]
    episodes = [
        EpisodeData(
            index=int(e),
            state=state[a:b],
            action=action[a:b],
            timestamp=ts[a:b],
            frame_index=fr_idx[a:b],
            success=None if success is None else success[a:b],
        )
        for e, a, b in zip(uniq, starts, bounds, strict=True)
    ]
    return Dataset(
        root=root,
        info=info,
        codebase_version=version,
        fps=float(fps),
        state_key=state_key,
        action_key=action_key,
        state_names=_feature_names(info, state_key),
        action_names=_feature_names(info, action_key),
        episodes=episodes,
        tags=_read_tags(root, info, notes),
        notes=notes,
    )

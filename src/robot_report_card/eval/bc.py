"""Tiny behavior-cloning policy: an MLP [state, environment_state] -> action − state, trained on CPU with torch.

Ported from ``docs/spikes/phase3_bc_eval.py`` (MLP 9→256→256→6, Adam 1e-3, 40 epochs, batch 256, MSE on
standardized targets, ``torch.manual_seed`` + a seeded permutation generator). A checkpoint directory holds
``model.pt`` (state dict) and ``rrc_policy.json`` (architecture, normalization, keys, training settings, dataset
fingerprint, filter, kept episodes, weights sha256). Same seed + same machine => identical weights (any thread count).

Note for library callers: ``threads`` (training) and the ``bc:`` loader call ``torch.set_num_threads``, which is
process-global; the CLI runs one command per process, so this only matters when embedding.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_report_card import __version__
from robot_report_card.eval.policies import ENV_STATE, STATE, PolicySpecError, ResetContext
from robot_report_card.hints import install_hint

ACTION = "action"
POLICY_FILE = "rrc_policy.json"
WEIGHTS_FILE = "model.pt"
KEEP_CHOICES = ("all", "quality-ok", "success", "ok-and-success")
SIM_OBS_DIM, SIM_ACTION_DIM = 9, 6


class TrainError(ValueError):
    """A user-facing training problem (bad filter, missing columns, output dir exists...)."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TrainError(f"bc policies need torch: {install_hint('eval')}") from e
    return torch


def build_mlp(torch: Any, x_mu: Any, x_sd: Any, y_mu: Any, y_sd: Any, hidden: int = 256) -> Any:
    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, v in {"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd}.items():
                self.register_buffer(name, torch.as_tensor(np.asarray(v), dtype=torch.float32))
            n_in, n_out = len(x_mu), len(y_mu)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_in, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, n_out),
            )

        def forward(self, x: Any) -> Any:  # raw obs -> raw (action - state)
            return self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu

    return MLP()


def weights_sha256(model: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---- data ----------------------------------------------------------------------------------------------------------
@dataclass
class TrainingData:
    x: np.ndarray  # (N, D + E) float32: [state, environment_state]
    y: np.ndarray  # (N, D) float32: action - state
    episodes: list[int]  # kept episode indices
    dataset: dict[str, Any]


def read_training_arrays(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """state, environment_state, action, episode_index per frame (float32, sorted by episode then frame)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from robot_report_card.score.reader import ReaderError, _to_2d, layout_for

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise TrainError(f"{root} is not a LeRobot dataset (no meta/info.json)")
    info = json.loads(info_path.read_text())
    try:
        files = sorted(root.glob(layout_for(str(info.get("codebase_version")))))
    except ReaderError as e:
        raise TrainError(str(e)) from e
    cols = [STATE, ENV_STATE, ACTION, "episode_index", "frame_index"]
    if not files:
        raise TrainError(f"no data files under {root}")
    names = pq.read_schema(files[0]).names
    missing = [c for c in cols if c not in names]
    if missing:
        raise TrainError(f"dataset lacks column(s) {missing}; bc needs {STATE}, {ENV_STATE} and {ACTION}")
    t = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    ep = np.asarray(t.column("episode_index").to_numpy(), np.int64)
    fr = np.asarray(t.column("frame_index").to_numpy(), np.int64)
    order = np.lexsort((fr, ep))
    arrays = {c: _to_2d(t.column(c), c)[order].astype(np.float32) for c in (STATE, ENV_STATE, ACTION)}
    arrays["episode_index"] = ep[order]
    fingerprint = {
        "path": str(root.resolve()),
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
    }
    return fingerprint, arrays


def select_episodes(root: Path, keep: str, score_json: Path | None) -> tuple[list[int] | None, dict[str, Any]]:
    """Episode indices to train on (None = all) and a description of the filter for rrc_policy.json.

    quality-ok: `rrc score` quality == ok. success: outcome_label if present, else outcome_sim (last-frame
    next.success). Episodes with unknown outcome make the success filters refuse (no guessing).
    """
    if keep not in KEEP_CHOICES:
        raise TrainError(f"--keep must be one of {KEEP_CHOICES}")
    desc: dict[str, Any] = {"keep": keep, "score_json": None}
    if keep == "all":
        return None, desc
    from robot_report_card.score.reader import ReaderError, read_dataset

    try:
        ds = read_dataset(root)
    except ReaderError as e:
        raise TrainError(str(e)) from e
    all_idx = [e.index for e in ds.episodes]
    keep_set = set(all_idx)
    if keep in ("quality-ok", "ok-and-success"):
        if score_json is None:
            raise TrainError(f"--keep {keep} needs --score-json (the report from `rrc score {root}`)")
        try:
            report = json.loads(score_json.read_text())
        except (OSError, ValueError) as e:
            raise TrainError(f"can't read --score-json {score_json}: {e}") from e
        rep_ds = report.get("dataset", {})
        if Path(str(rep_ds.get("path"))).resolve() != root.resolve() or rep_ds.get("total_frames") != ds.num_frames:
            raise TrainError(f"--score-json {score_json} was made for {rep_ds.get('path')}, not {root}")
        quality = {int(e["episode_index"]): e["quality"] for e in report.get("episodes", [])}
        if set(quality) != set(all_idx):
            raise TrainError(f"--score-json {score_json} doesn't cover this dataset's episodes")
        keep_set &= {i for i, q in quality.items() if q == "ok"}
        desc["score_json"] = str(score_json.resolve())
        desc["score_json_sha256"] = hashlib.sha256(score_json.read_bytes()).hexdigest()
    if keep in ("success", "ok-and-success"):
        from robot_report_card.score.engine import outcome_fields

        outcome = {}
        for e in ds.episodes:
            sim, label = outcome_fields(ds, e.index, e.success)
            outcome[e.index] = (label == "success") if label is not None else sim
        unknown = [i for i, v in outcome.items() if v is None]
        if unknown:
            raise TrainError(
                f"--keep {keep}: {len(unknown)} episode(s) have no outcome evidence (no label in rrc_tags.json and "
                f"no next.success), e.g. {unknown[:5]}; use --keep all or quality-ok"
            )
        keep_set &= {i for i, v in outcome.items() if v}
        desc["outcome_source"] = "label if present, else last-frame next.success"
    kept = sorted(keep_set)
    if not kept:
        raise TrainError(f"--keep {keep} leaves no episodes to train on")
    return kept, desc


# ---- training --------------------------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    hidden: int = 256
    threads: int | None = None


def train_bc(
    dataset: Path,
    out: Path,
    keep: str = "all",
    score_json: Path | None = None,
    cfg: TrainConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    torch = _torch()
    dataset, out = Path(dataset), Path(out)
    _check_out(out, overwrite)
    fingerprint, arrays = read_training_arrays(dataset)
    kept, filt = select_episodes(dataset, keep, score_json)
    mask = np.ones(len(arrays["episode_index"]), bool) if kept is None else np.isin(arrays["episode_index"], kept)
    kept_eps = sorted(set(arrays["episode_index"][mask].tolist()))
    x = np.concatenate([arrays[STATE], arrays[ENV_STATE]], axis=1)[mask]
    y = (arrays[ACTION] - arrays[STATE])[mask]
    if cfg.threads is not None:
        torch.set_num_threads(cfg.threads)
    t0 = time.perf_counter()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = build_mlp(torch, x.mean(0), x.std(0) + 1e-6, y.mean(0), y.std(0) + 1e-6, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    ysd = model.y_sd
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            loss = (((model(xt[idx]) - yt[idx]) / ysd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    digest = weights_sha256(model)
    meta = {
        "schema_version": 1,
        "kind": "bc-mlp",
        "rrc_version": __version__,
        "arch": {"inputs": int(x.shape[1]), "hidden": [cfg.hidden, cfg.hidden], "outputs": int(y.shape[1]),
                 "activation": "relu", "target": "action - observation.state"},
        "obs_keys": [STATE, ENV_STATE],
        "action_key": ACTION,
        "normalization": {"x_mu": x.mean(0).tolist(), "x_sd": (x.std(0) + 1e-6).tolist(),
                          "y_mu": y.mean(0).tolist(), "y_sd": (y.std(0) + 1e-6).tolist()},
        "train": {"seed": cfg.seed, "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
                  "optimizer": "adam", "threads": torch.get_num_threads(), "frames": int(len(x)),
                  "seconds": round(train_s, 3), "torch": torch.__version__},
        "dataset": fingerprint,
        "filter": filt,
        "kept_episodes": kept_eps,
        "weights_sha256": digest,
    }  # fmt: skip
    from robot_report_card.atomic import atomic_write_json

    # write the new checkpoint next to --out, then swap it in: a failed run never loses the old checkpoint
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.new-", dir=out.parent))
    try:
        torch.save(model.state_dict(), tmp / WEIGHTS_FILE)
        atomic_write_json(tmp / POLICY_FILE, meta)
        _swap_in(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return meta


def _swap_in(new: Path, out: Path) -> None:
    if not out.exists():
        new.rename(out)
        return
    old = out.with_name(f".{out.name}.old-{os.getpid()}")
    out.rename(old)
    try:
        new.rename(out)
    except BaseException:
        old.rename(out)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _check_out(out: Path, overwrite: bool) -> None:
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise TrainError(f"--out {out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()):
        if not overwrite:
            raise TrainError(f"{out} already exists; pass --overwrite to replace the checkpoint (or pick a new --out)")
        if not (out / POLICY_FILE).is_file():
            raise TrainError(f"{out} is not an rrc checkpoint (no {POLICY_FILE}); refusing to overwrite it")


# ---- the `bc:` policy ------------------------------------------------------------------------------------
class BCPolicy:
    kind = "bc"

    def __init__(self, spec: str, path: Path, threads: int | None = None) -> None:
        torch = _torch()
        meta_path = path / POLICY_FILE
        if not meta_path.is_file() or not (path / WEIGHTS_FILE).is_file():
            raise PolicySpecError(
                f"{spec}: {path} is not an rrc bc checkpoint (needs {POLICY_FILE} and {WEIGHTS_FILE})"
            )
        try:
            meta = json.loads(meta_path.read_text())
            norm = meta["normalization"]
            self.model = build_mlp(
                torch, norm["x_mu"], norm["x_sd"], norm["y_mu"], norm["y_sd"], meta["arch"]["hidden"][0]
            )
            self.model.load_state_dict(torch.load(path / WEIGHTS_FILE, map_location="cpu", weights_only=True))
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, EOFError, pickle.UnpicklingError) as e:
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]  # torch appends long advice
            raise PolicySpecError(f"{spec}: can't load checkpoint {path}: {first}") from e
        arch = meta.get("arch", {})
        if arch.get("inputs") != SIM_OBS_DIM or arch.get("outputs") != SIM_ACTION_DIM:
            raise PolicySpecError(
                f"{spec}: checkpoint maps {arch.get('inputs')} inputs to {arch.get('outputs')} outputs; so101_reach "
                f"needs {SIM_OBS_DIM} ([state 6, target xyz 3]) -> {SIM_ACTION_DIM}"
            )
        if weights_sha256(self.model) != meta.get("weights_sha256"):
            raise PolicySpecError(f"{spec}: weights in {path} don't match rrc_policy.json (weights_sha256)")
        if threads is not None:
            torch.set_num_threads(threads)
        self.model.eval()
        self.spec, self.meta, self._torch = spec, meta, torch

    def metadata(self) -> dict[str, Any]:
        m = self.meta
        return {
            "spec": self.spec,
            "kind": self.kind,
            "weights_sha256": m["weights_sha256"],
            "train": m["train"],
            "dataset": m["dataset"],
            "filter": m["filter"],
            "kept_episodes": len(m["kept_episodes"]),
            "kept_episode_indices": list(m["kept_episodes"]),  # lets `rrc report` apply its rules to this subset
        }

    def reset(self, seed: np.random.SeedSequence, ctx: ResetContext) -> None:
        return None  # deterministic, stateless

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        x = np.concatenate([obs[STATE], obs[ENV_STATE]])[None]
        with torch.no_grad():
            delta = self.model(torch.as_tensor(x, dtype=torch.float32))[0].numpy().astype(np.float64)
        return obs[STATE] + delta


def load_bc_policy(spec: str, path: Path, threads: int | None = None) -> BCPolicy:
    try:
        return BCPolicy(spec, path, threads)
    except TrainError as e:
        raise PolicySpecError(str(e)) from e

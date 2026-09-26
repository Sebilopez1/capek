"""Per-frame feature spec shared by the recorder (session ``.npz``) and the LeRobot exporter.

The format mirrors LeRobot's ``features`` dict: ``{key: {"dtype", "shape", "names"}}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

STATE = "observation.state"
ENV_STATE = "observation.environment_state"
ACTION = "action"
REWARD = "next.reward"
SUCCESS = "next.success"
FEATURE_KEYS: tuple[str, ...] = (STATE, ENV_STATE, ACTION, REWARD, SUCCESS)

FeatureSpec = dict[str, dict[str, Any]]


def make_features(
    state_names: Sequence[str], env_state_names: Sequence[str], action_names: Sequence[str]
) -> FeatureSpec:
    """Build the phase-1 feature spec (float32 vectors, ``(1,)`` reward, ``(1,)`` bool success)."""
    return {
        STATE: {"dtype": "float32", "shape": [len(state_names)], "names": list(state_names)},
        ENV_STATE: {"dtype": "float32", "shape": [len(env_state_names)], "names": list(env_state_names)},
        ACTION: {"dtype": "float32", "shape": [len(action_names)], "names": list(action_names)},
        REWARD: {"dtype": "float32", "shape": [1], "names": None},
        SUCCESS: {"dtype": "bool", "shape": [1], "names": None},
    }


def numpy_dtype(spec: dict[str, Any]) -> np.dtype:
    return np.dtype(spec["dtype"])


def validate_arrays(arrays: dict[str, np.ndarray], features: FeatureSpec) -> int:
    """Check keys, dtypes and per-frame shapes; return the common frame count."""
    if set(arrays) != set(features):
        raise ValueError(f"feature keys {sorted(arrays)} != expected {sorted(features)}")
    lengths = set()
    for key, spec in features.items():
        arr = arrays[key]
        if arr.dtype != numpy_dtype(spec):
            raise ValueError(f"{key}: dtype {arr.dtype} != {spec['dtype']}")
        if arr.ndim != 1 + len(spec["shape"]) or list(arr.shape[1:]) != list(spec["shape"]):
            raise ValueError(f"{key}: shape {arr.shape} != (T, {', '.join(map(str, spec['shape']))})")
        lengths.add(arr.shape[0])
    if len(lengths) != 1:
        raise ValueError(f"features have different frame counts: {sorted(lengths)}")
    return lengths.pop()

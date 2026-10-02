"""``so101_reach``: SO-101 arm in MuJoCo, move the gripper tip to a reachable target point (research brief §3).

- Model: vendored ``so101_new_calib.xml``; 6 position actuators (kp=17.8) on joints ``"1"``..``"6"``.
- Timing: fps=30, 17 physics substeps per frame, ``timestep = 1/(30*17)`` so physics matches timestamps (G8).
- Reset: start at ``qpos = 0``; sample ``q* ~ U(0.6 * jnt_range)``; target = FK(q*) at the ``gripper`` site.
- Action: joint position targets (rad), clipped to the joint ranges. State: ``qpos`` (rad). Env state: target xyz (m).
- Reward: ``-tip_error``; success: tip error < 2 cm (evaluated after each step; episode success = final frame).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from capek.features import FeatureSpec, make_features
from capek.sim.assets import so101_xml_path
from capek.sim.base import Observation, SeedLike, StepResult

JOINT_NAMES: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_NAMES: list[str] = [f"{j}.pos" for j in JOINT_NAMES]
ENV_STATE_NAMES: list[str] = ["target_x", "target_y", "target_z"]
TARGET_RANGE_FRACTION = 0.6


class SO101Reach:
    env_id = "so101_reach"
    fps = 30
    substeps = 17
    success_threshold_m = 0.02

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(so101_xml_path()))
        self.model.opt.timestep = 1.0 / (self.fps * self.substeps)
        assert abs(self.substeps * self.model.opt.timestep - 1.0 / self.fps) < 1e-12, "physics grid != frame grid"
        self.data = mujoco.MjData(self.model)
        assert self.model.nu == 6 and self.model.njnt == 6
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self.action_low = self.model.jnt_range[:, 0].copy()
        self.action_high = self.model.jnt_range[:, 1].copy()
        self.features: FeatureSpec = make_features(STATE_NAMES, ENV_STATE_NAMES, STATE_NAMES)
        self._target = np.zeros(3)

    # ---- helpers ---------------------------------------------------------------------------------------------
    def _observation(self) -> Observation:
        return Observation(state=self.data.qpos.copy(), env_state=self._target.copy())

    def forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
        """Gripper-site position (m) for joint config ``qpos``; leaves the sim state reset."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        xyz = self.data.site_xpos[self._site].copy()
        mujoco.mj_resetData(self.model, self.data)
        return xyz

    def gravity_offset(self, qpos: np.ndarray) -> np.ndarray:
        """Position-target offset (rad) that cancels gravity at static config ``qpos``: bias_torque / kp."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        kp = self.model.actuator_gainprm[:, 0]
        return d.qfrc_bias.copy() / kp

    # ---- EnvAdapter --------------------------------------------------------------------------------------------
    def reset(self, seed: SeedLike) -> tuple[Observation, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        target_qpos = rng.uniform(TARGET_RANGE_FRACTION * self.action_low, TARGET_RANGE_FRACTION * self.action_high)
        self._target = self.forward_kinematics(target_qpos)  # also resets data to qpos = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"target_qpos": target_qpos, "target_xyz": self._target.copy()}

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,):
            raise ValueError(f"action shape {action.shape} != ({self.model.nu},)")
        if not np.all(np.isfinite(action)):
            raise ValueError("non-finite action")
        applied = np.clip(action, self.action_low, self.action_high)
        self.data.ctrl[:] = applied
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)):
            raise FloatingPointError("simulation diverged (non-finite qpos)")
        err = self.tip_error()
        return StepResult(self._observation(), applied, -err, err < self.success_threshold_m)

    def tip_error(self) -> float:
        return float(np.linalg.norm(self.data.site_xpos[self._site] - self._target))

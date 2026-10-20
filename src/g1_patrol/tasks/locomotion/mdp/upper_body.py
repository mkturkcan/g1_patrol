"""Random upper-body motion for training a walker that keeps its balance while the arms work.

The walker's policy drives the legs only; the waist, arms and fingers follow targets from elsewhere (an arm controller
when picking things up). In training, this action term moves them along random smooth trajectories, so that the
walker learns to balance whatever the upper body does: arms at rest, holding something in front of the hip or down by
the side, raised forward (over a basket, or both arms up as a guard), reaching down to the ground (more often while the
posture command asks for a squat), or anywhere within a broad range.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

UPPER_BODY_JOINTS = [
    "torso_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
    ".*_(zero|one|two|three|four|five|six)_joint",
]
"""Joints of the G1 that the walker does not drive."""

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")
# (low, high) per arm joint for the right arm; the left arm mirrors roll and yaw. Shoulder pitch: negative raises the
# arm forward. Elbow pitch: 0 bends the forearm forward at a right angle to the upper arm, about 1.57 straightens it.
ARM_MODES = {
    "reach": ((-0.6, 1.0), (-1.3, -0.3), (-1.8, 1.4), (0.5, 2.0), (-1.7, 1.7)),  # down to the ground in a squat
    "raise": ((-1.8, -0.8), (-0.4, 0.1), (-0.5, 0.5), (0.0, 1.0), (-0.5, 0.5)),  # upper arm forward, forearm up
    "carry": ((-0.6, 0.3), (-0.4, 0.0), (-0.3, 0.3), (1.0, 1.8), (-0.5, 0.5)),  # arm down by the side, nearly straight
    "hang": ((-0.2, 0.4), (-0.45, -0.15), (-0.3, 0.3), (0.0, 0.5), (-0.3, 0.3)),  # forearm forward at the hip, a little out
    "random": ((-2.0, 1.2), (-1.5, 0.3), (-1.5, 1.5), (-0.2, 2.5), (-1.5, 1.5)),
    "guard": ((-1.5, -0.9), (-0.1, 0.3), (-0.4, 0.4), (1.3, 1.9), (-0.4, 0.4)),  # both arms raised forward: "stop"
}


class UpperBodyMotionAction(ActionTerm):
    """Moves the upper-body joints towards random targets at a limited speed. Takes no policy inputs."""

    cfg: UpperBodyMotionActionCfg

    def __init__(self, cfg: UpperBodyMotionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._names = names
        n, dev = self.num_envs, self.device
        self._default = self._asset.data.default_joint_pos.torch[:, self._ids].clone()
        lim = self._asset.data.soft_joint_pos_limits.torch[0, self._ids]
        self._lo, self._hi = lim[:, 0], lim[:, 1]
        self.target = self._default.clone()
        self.current = self._default.clone()
        self._timer = torch.zeros(n, device=dev)
        self._empty = torch.zeros(n, 0, device=dev)
        index = {name: k for k, name in enumerate(names)}
        self._arm = {
            side: torch.tensor([index[f"{side}_{j}_joint"] for j in ARM], device=dev) for side in ("left", "right")
        }
        self._waist = index["torso_joint"]
        self._fingers = {
            side: torch.tensor([k for name, k in index.items() if name.startswith(side) and name.split("_")[1] in
                                ("zero", "one", "two", "three", "four", "five", "six")], device=dev)
            for side in ("left", "right")
        }
        # mirror the right arm's ranges for the left arm (roll and yaw change sign)
        mirror = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0], device=dev)
        self._modes = {}
        for mode, ranges in ARM_MODES.items():
            r = torch.tensor(ranges, device=dev)
            left = torch.stack([r[:, 0] * mirror, r[:, 1] * mirror], dim=1).sort(dim=1).values
            self._modes[mode] = {"right": r, "left": left}

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.target[ids] = self._default[ids]
        self.current[ids] = self._default[ids]
        self._timer[ids] = 0.5 + torch.rand_like(self._timer[ids])

    def _sample(self, env_ids: torch.Tensor):
        k, dev, cfg = len(env_ids), self.device, self.cfg
        target = self._default[env_ids] + cfg.rest_noise * (2 * torch.rand(k, len(self._ids), device=dev) - 1)
        low = torch.zeros(k, dtype=torch.bool, device=dev)
        if cfg.posture_command is not None:
            command = self._env.command_manager.get_command(cfg.posture_command)
            low = command[env_ids, 3] < cfg.squat_height
        u = torch.rand(k, device=dev)
        # one arm works, the other rests (or carries)
        mode = torch.where(
            low,
            torch.where(u < 0.7, 0, torch.where(u < 0.8, 4, 3)),  # reach / rest / random
            torch.where(u < 0.35, 4, torch.where(u < 0.5, 2, torch.where(u < 0.65, 5, torch.where(u < 0.75, 1, 3)))),
        )  # standing or walking: rest / carry / hang / raise / random
        guard = torch.zeros(k, dtype=torch.bool, device=dev)
        if cfg.posture_command is not None:
            term = self._env.command_manager.get_term(cfg.posture_command)
            if hasattr(term, "is_guard_env"):
                guard = term.is_guard_env[env_ids] & (torch.rand(k, device=dev) < 0.8)
        mode = torch.where(guard, 6, mode)
        right = torch.rand(k, device=dev) < 0.5
        for side in ("left", "right"):  # guard: both arms
            r = self._modes["guard"][side]
            if guard.any():
                sample = r[:, 0] + (r[:, 1] - r[:, 0]) * torch.rand(int(guard.sum()), len(ARM), device=dev)
                rows = target[guard]
                rows[:, self._arm[side]] = sample
                target[guard] = rows
        for m, name in ((0, "reach"), (1, "raise"), (2, "carry"), (3, "random"), (5, "hang")):
            sel = mode == m
            for side, use in (("right", right), ("left", ~right)):
                s = sel & use
                if not s.any():
                    continue
                r = self._modes[name][side]
                sample = r[:, 0] + (r[:, 1] - r[:, 0]) * torch.rand(int(s.sum()), len(ARM), device=dev)
                rows = target[s]
                rows[:, self._arm[side]] = sample
                target[s] = rows
        # the waist turns a little, more when reaching
        reach = mode == 0
        waist = cfg.waist_range * (2 * torch.rand(k, device=dev) - 1) * torch.where(reach, 1.0, 0.5)
        target[:, self._waist] = waist
        # fingers: open, closed (a grasp) or in between
        for side in ("left", "right"):
            f = self._fingers[side]
            closed = torch.rand(k, 1, device=dev)
            span = self._lo[f] + (self._hi[f] - self._lo[f]) * torch.rand(k, len(f), device=dev)
            target[:, f] = self._default[env_ids][:, f] * (1 - closed) + span * closed
        self.target[env_ids] = torch.maximum(torch.minimum(target, self._hi), self._lo)
        self._timer[env_ids] = cfg.interval[0] + (cfg.interval[1] - cfg.interval[0]) * torch.rand(k, device=dev)

    def apply_actions(self):
        # called at every physics step, or once per control step when the physics engine runs the substeps (Newton)
        env = self._env
        dt = env.physics_dt * (env.cfg.decimation if getattr(env, "_physics_handles_decimation", False) else 1)
        if self.cfg.enabled:
            self._timer -= dt
            due = (self._timer <= 0.0).nonzero().flatten()
            if len(due):
                self._sample(due)
        step = self.cfg.speed * dt
        self.current += (self.target - self.current).clamp(-step, step)
        self._asset.set_joint_position_target_index(target=self.current, joint_ids=self._ids)


@configclass
class UpperBodyMotionActionCfg(ActionTermCfg):
    class_type: type = UpperBodyMotionAction
    joint_names: list[str] = MISSING
    enabled: bool = True
    """Random motion; when disabled the upper body holds its default pose."""
    interval: tuple[float, float] = (1.0, 3.0)
    """Seconds between new targets."""
    speed: float = 2.5
    """Largest joint speed (rad/s) towards the target."""
    rest_noise: float = 0.1
    waist_range: float = 0.5
    posture_command: str | None = "base_velocity"
    """Command whose pelvis height (column 3) makes reaching down more likely below ``squat_height``."""
    squat_height: float = 0.6

"""Posture commands and rewards for a walker that also squats and leans forward (to pick things up from the ground).

The command extends Isaac Lab's base-velocity command with a pelvis height above the ground and a forward lean of the
pelvis (and torso, which the G1 turns only about the vertical axis). Postures are held while standing: a share of the
command samples asks for zero velocity and a posture, which is approached at a limited rate and released back to the
walking posture after a few seconds, as when the robot stops, crouches to pick something up and stands up again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster


class VelocityPostureCommand(UniformVelocityCommand):
    """Base velocity (forward, lateral, yaw rate) plus pelvis height (m above the ground) and forward lean (rad).

    The command is ``(vx, vy, wz, height, lean)``; height and lean move towards their targets at ``height_rate`` and
    ``lean_rate`` per second, so a posture is entered and left smoothly.
    """

    cfg: VelocityPostureCommandCfg

    def __init__(self, cfg: VelocityPostureCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self.posture_target = torch.tensor([cfg.nominal_height, 0.0], device=self.device).repeat(n, 1)
        self.posture = self.posture_target.clone()
        self.is_posture_env = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.is_guard_env = torch.zeros(n, dtype=torch.bool, device=self.device)
        """Braced stance (lowered pelvis, forearms raised as a guard): defending against an attacker."""
        self.hold_time = torch.zeros(n, device=self.device)
        self.metrics["error_height"] = torch.zeros(n, device=self.device)
        self.metrics["error_lean"] = torch.zeros(n, device=self.device)
        self._error_height_sum = torch.zeros(n, device=self.device)
        self._error_lean_sum = torch.zeros(n, device=self.device)
        self.cfg.element_names = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "pelvis_height", "lean"]

    @property
    def command(self) -> torch.Tensor:
        """(vx, vy, wz, pelvis height, lean). Shape is (num_envs, 5)."""
        return torch.cat([self.vel_command_b, self.posture], dim=1)

    def set_posture(self, height: torch.Tensor, lean: torch.Tensor, env_ids=None):
        """Externally set the posture target (e.g. by a controller that picks things up)."""
        ids = slice(None) if env_ids is None else env_ids
        self.posture_target[ids, 0] = height
        self.posture_target[ids, 1] = lean

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        denom = self._step_count[ids].clamp_min(1.0)
        self.metrics["error_height"][ids] = self._error_height_sum[ids] / denom
        self.metrics["error_lean"][ids] = self._error_lean_sum[ids] / denom
        self._error_height_sum[ids] = 0.0
        self._error_lean_sum[ids] = 0.0
        extras = super().reset(env_ids)
        # every episode starts standing in the walking posture
        self.posture[ids, 0] = self.cfg.nominal_height
        self.posture[ids, 1] = 0.0
        return extras

    def _update_metrics(self):
        super()._update_metrics()
        height, lean = measured_posture(self._env, self.robot, self.cfg.height_sensor_name)
        self._error_height_sum += (height - self.posture[:, 0]).abs()
        self._error_lean_sum += (lean - self.posture[:, 1]).abs()

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = len(env_ids)
        if k == 0:
            return
        cfg, dev = self.cfg, self.device
        if cfg.slow_ratio > 0.0:  # slow walking, or turning on the spot, with a yaw-rate command (no heading target)
            slow = env_ids[torch.rand(k, device=dev) < cfg.slow_ratio]
            m = len(slow)
            if m:
                turn = torch.rand(m, device=dev) < 0.5
                vx = (2 * torch.rand(m, device=dev) - 1) * torch.where(turn, 0.1, cfg.slow_speed)
                vy = (2 * torch.rand(m, device=dev) - 1) * torch.where(turn, 0.0, cfg.slow_lateral)
                lo, hi = cfg.turn_rate
                rate = (lo + (hi - lo) * torch.rand(m, device=dev)) * torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
                wz = torch.where(turn, rate, (2 * torch.rand(m, device=dev) - 1) * hi)
                self.vel_command_b[slow] = torch.stack([vx, vy, wz], dim=1)
                if cfg.heading_command:
                    self.is_heading_env[slow] = False
        if not cfg.sample_postures:
            return
        posture = torch.rand(k, device=dev) < cfg.posture_ratio
        deep = torch.rand(k, device=dev) < cfg.deep_ratio
        h_lo, h_hi = cfg.height_range
        # general postures: lower pelvis allows more lean (the hips go back as the torso goes forward)
        h = h_lo + (h_hi - h_lo) * torch.rand(k, device=dev)
        frac = ((cfg.nominal_height - h) / (cfg.nominal_height - h_lo) * 1.3).clamp(0.25, 1.0)
        lean = cfg.lean_range[1] * frac * torch.rand(k, device=dev)
        # pick-like postures: deep squat and strong lean
        h = torch.where(deep, cfg.deep_height[0] + (cfg.deep_height[1] - cfg.deep_height[0]) * torch.rand(k, device=dev), h)
        lean = torch.where(deep, cfg.deep_lean[0] + (cfg.deep_lean[1] - cfg.deep_lean[0]) * torch.rand(k, device=dev), lean)
        walk_h = cfg.nominal_height + cfg.walk_height_noise * (2 * torch.rand(k, device=dev) - 1)
        # braced stance (held, or while backing off and walking away): pelvis lowered a little, slight lean
        guard = torch.rand(k, device=dev) < cfg.guard_ratio
        guard_h = cfg.guard_height[0] + (cfg.guard_height[1] - cfg.guard_height[0]) * torch.rand(k, device=dev)
        guard_lean = cfg.guard_lean[1] * torch.rand(k, device=dev)
        h = torch.where(guard & posture, guard_h, h)
        lean = torch.where(guard & posture, guard_lean, lean)
        walk_h = torch.where(guard & ~posture, guard_h.clamp(min=0.64), walk_h)
        walk_lean = torch.where(guard & ~posture, 0.5 * guard_lean, torch.zeros_like(lean))
        self.posture_target[env_ids, 0] = torch.where(posture, h, walk_h)
        self.posture_target[env_ids, 1] = torch.where(posture, lean, walk_lean)
        self.is_posture_env[env_ids] = posture
        self.is_guard_env[env_ids] = guard
        # a posture is held (standing still) and then released after a few seconds
        self.is_standing_env[env_ids] |= posture
        self.hold_time[env_ids] = cfg.hold_time[0] + (cfg.hold_time[1] - cfg.hold_time[0]) * torch.rand(k, device=dev)

    def _update_command(self):
        super()._update_command()
        dt = self._env.step_dt
        if self.cfg.sample_postures:
            self.hold_time -= dt
            release = self.is_posture_env & (self.hold_time <= 0.0)
            self.posture_target[release, 0] = self.cfg.nominal_height
            self.posture_target[release, 1] = 0.0
            self.is_posture_env &= ~release
            self.is_guard_env &= ~release
        # move towards the target at a limited rate
        rate = torch.tensor([self.cfg.height_rate, self.cfg.lean_rate], device=self.device) * dt
        self.posture += (self.posture_target - self.posture).clamp(-rate, rate)


@configclass
class VelocityPostureCommandCfg(UniformVelocityCommandCfg):
    class_type: type = VelocityPostureCommand

    nominal_height: float = 0.72
    """Pelvis height (m above the ground) of the walking posture."""
    walk_height_noise: float = 0.02
    sample_postures: bool = True
    """Sample postures in training; a controller sets them otherwise (``set_posture``)."""
    posture_ratio: float = 0.3
    """Share of command samples that ask the robot to stand still in a posture."""
    deep_ratio: float = 0.5
    """Share of those postures that are pick-like: a deep squat with a strong forward lean."""
    height_range: tuple[float, float] = (0.40, 0.72)
    lean_range: tuple[float, float] = (0.0, 1.2)
    deep_height: tuple[float, float] = (0.40, 0.50)
    deep_lean: tuple[float, float] = (0.8, 1.2)
    guard_ratio: float = 0.25
    """Share of command samples in a braced stance (held, or walking with it)."""
    guard_height: tuple[float, float] = (0.58, 0.68)
    guard_lean: tuple[float, float] = (0.0, 0.35)
    hold_time: tuple[float, float] = (2.0, 6.0)
    """Seconds a posture is held before the target returns to the walking posture."""
    height_rate: float = 0.4
    lean_rate: float = 1.2
    height_sensor_name: str = "height_scanner"
    slow_ratio: float = 0.0
    """Share of command samples that are slow walking (up to ``slow_speed`` forward or backward and ``slow_lateral``
    sideways, any yaw rate) or, half of them, turning on the spot (``turn_rate``), all with a yaw-rate command instead
    of a heading target: what a navigation policy asks for in a crowd."""
    slow_speed: float = 0.3
    slow_lateral: float = 0.15
    turn_rate: tuple[float, float] = (0.4, 1.5)


def measured_posture(env, asset, sensor_name: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pelvis height above the ground below the robot (mean of the height scan's finite hits) and forward lean of the
    pelvis (rad, from the direction of gravity in the pelvis frame)."""
    z = asset.data.root_pos_w.torch[:, 2]
    if sensor_name is not None and sensor_name in env.scene.sensors:
        sensor: RayCaster = env.scene.sensors[sensor_name]
        hits = sensor.data.ray_hits_w.torch[..., 2]
        finite = torch.isfinite(hits)
        ground = torch.where(finite, hits, torch.zeros_like(hits)).sum(1) / finite.sum(1).clamp(min=1)
        z = z - ground
    gravity = quat_apply_inverse(asset.data.root_quat_w.torch, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
    lean = torch.atan2(gravity[:, 0], -gravity[:, 2])
    return z, lean


def track_posture_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_std: float = 0.05,
    lean_std: float = 0.15,
    sensor_name: str = "height_scanner",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-(height error / height_std)^2 - (lean error / lean_std)^2)."""
    command = env.command_manager.get_command(command_name)
    height, lean = measured_posture(env, env.scene[asset_cfg.name], sensor_name)
    error = ((height - command[:, 3]) / height_std) ** 2 + ((lean - command[:, 4]) / lean_std) ** 2
    return torch.exp(-error)


def posture_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    component: int,
    sensor_name: str = "height_scanner",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error of the pelvis height (``component`` 0, m) or of the forward lean (1, rad)."""
    command = env.command_manager.get_command(command_name)
    measured = measured_posture(env, env.scene[asset_cfg.name], sensor_name)[component]
    return (measured - command[:, 3 + component]) ** 2


def sideways_tilt_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared sideways component of gravity in the pelvis frame (roll); forward lean is commanded."""
    asset = env.scene[asset_cfg.name]
    gravity = quat_apply_inverse(asset.data.root_quat_w.torch, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
    return gravity[:, 1] ** 2


def planted_feet(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """While standing (no velocity command): feet off the ground plus foot speed (m/s), so that postures are taken
    with both feet planted."""
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(command[:, :2], dim=1) < command_threshold) & (command[:, 2].abs() < command_threshold)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    airborne = (sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] <= 0.0).float().sum(dim=1)
    asset = env.scene[asset_cfg.name]
    speed = torch.linalg.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1).sum(dim=1)
    return (airborne + speed) * standing


__all__ = [
    "VelocityPostureCommand",
    "VelocityPostureCommandCfg",
    "measured_posture",
    "planted_feet",
    "posture_error_l2",
    "sideways_tilt_l2",
    "track_posture_exp",
]


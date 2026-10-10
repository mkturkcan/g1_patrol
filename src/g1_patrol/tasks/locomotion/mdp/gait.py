"""Gait-shaping terms for an upright, periodic humanoid walk.

The gait clock is derived from the simulation's physics step counter rather than the episode step counter, so the
same phase is observed whether the walker runs in its own environment or as a frozen low-level policy inside another
task with a different control rate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _phase(env: ManagerBasedEnv, period: float) -> float:
    """Gait phase in [0, 1), shared by all environments."""
    sim_time = env.sim.get_physics_step_count() * env.sim.get_physics_dt()
    return (sim_time / period) % 1.0


def gait_phase(env: ManagerBasedEnv, period: float = 0.8) -> torch.Tensor:
    """Gait clock observation: (sin, cos) of the phase. Shape (num_envs, 2)."""
    angle = 2.0 * math.pi * _phase(env, period)
    clock = torch.tensor([math.sin(angle), math.cos(angle)], device=env.device)
    return clock.expand(env.num_envs, 2)


def gait_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
    offsets: tuple[float, ...] = (0.0, 0.5),
    stance_ratio: float = 0.55,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Fraction of feet whose contact state matches the gait clock (stance for the first ``stance_ratio`` of each
    foot's cycle). When the command is near zero, rewards keeping both feet on the ground instead."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    leg_phase = (_phase(env, period) + torch.tensor(offsets, device=env.device)) % 1.0
    stance = (leg_phase < stance_ratio).unsqueeze(0)
    match = (contact == stance).float().mean(dim=1)
    command = env.command_manager.get_command(command_name)
    moving = (torch.linalg.norm(command[:, :2], dim=1) > command_threshold) | (command[:, 2].abs() > command_threshold)
    return torch.where(moving, match, contact.all(dim=1).float())


def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    std: float = 0.01,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """Reward lifting each moving foot to ``target_height`` above the lower foot (works on slopes and steps)."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    height = z - z.min(dim=1, keepdim=True).values
    speed = torch.linalg.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    error = torch.square(height - target_height) * torch.tanh(tanh_mult * speed)
    return torch.exp(-error.sum(dim=1) / std)


def body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tilt of one body (e.g. the torso) via the xy-components of gravity in its frame."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    return torch.sum(torch.square(quat_apply_inverse(quat, gravity)[:, :2]), dim=1)


def feet_lateral_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_distance: float = 0.18, max_distance: float = 0.40
) -> torch.Tensor:
    """Penalize feet closer than ``min_distance`` (crossing) or farther than ``max_distance`` (splayed) sideways."""
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    diff = torch.zeros(env.num_envs, 3, device=env.device)
    diff[:, :2] = feet[:, 0, :2] - feet[:, 1, :2]
    lateral = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), diff)[:, 1].abs()
    return (min_distance - lateral).clamp(min=0.0) + (lateral - max_distance).clamp(min=0.0)

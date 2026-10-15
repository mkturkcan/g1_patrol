"""Gait-shaping terms for an upright, periodic humanoid walk.

The gait clock is derived from the simulation's physics step counter rather than the episode step counter, so the
same phase is observed whether the walker runs in its own environment or as a frozen low-level policy inside another
task with a different control rate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mechanical power (W) of the selected joints, sum of |torque x joint velocity|."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def feet_impact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500.0) -> torch.Tensor:
    """Vertical contact force (kN) of the feet above ``threshold`` (N): stomping."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].abs()
    return torch.sum((force - threshold).clamp(min=0.0), dim=1) / 1000.0


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float = 3.0) -> torch.Tensor:
    """1 if a foot is pushed mostly sideways (hitting a curb edge or step riser with its toe or side)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    horizontal = torch.linalg.norm(force[..., :2], dim=-1)
    return ((horizontal > ratio * force[..., 2].abs()) & (horizontal > 5.0)).any(dim=1).float()


class action_second_difference(ManagerTermBase):
    """Squared second difference of the actions, ||a_t - 2 a_{t-1} + a_{t-2}||^2 (jerky joint targets)."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.prev_prev[ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action, prev = env.action_manager.action, env.action_manager.prev_action
        value = torch.sum(torch.square(action - 2.0 * prev + self.prev_prev), dim=1)
        fresh = env.episode_length_buf <= 2  # not enough history after a reset
        self.prev_prev[:] = prev
        return torch.where(fresh, torch.zeros_like(value), value)


class velocity_drift(ManagerTermBase):
    """Squared low-passed lateral velocity and yaw-rate tracking errors (heading frame, time constant
    ``time_constant``).

    The exponential tracking rewards are lenient about small sustained errors and penalize the gait's lateral sway as
    much as crabbing; filtering over about a gait cycle leaves the sustained part: sideways drift and heading drift
    under a straight command, which add up to tens of centimeters within a few steps in a narrow walkway. The forward
    error is left to the tracking reward (it is mostly acceleration after a command change), and each step's error is
    clipped to ``max_error`` so that a shove does not dominate the penalty for the following second.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.error = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.error[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        time_constant: float = 1.0,
        max_error: tuple[float, float] = (0.2, 0.3),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        vel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch)
        command = env.command_manager.get_command(command_name)
        limit = torch.tensor(max_error, device=env.device)
        error = torch.stack([vel[:, 1] - command[:, 1], asset.data.root_ang_vel_w.torch[:, 2] - command[:, 2]], dim=1)
        error = torch.maximum(torch.minimum(error, limit), -limit)
        alpha = min(env.step_dt / time_constant, 1.0)
        self.error.mul_(1.0 - alpha).add_(alpha * error)
        return torch.sum(self.error**2, dim=1)


class randomize_motor_strength(ManagerTermBase):
    """Scale the torque limit of each joint of the given explicit actuator groups by a factor drawn per robot and
    joint from ``strength_range`` (motors that are weaker than rated: heat, battery voltage, wear).

    Groups executed by Newton actuators keep the limit in their clamping component; Isaac Lab actuator models in
    ``actuator_effort_limit``.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        from isaaclab.actuators.newton import read_group_parameter

        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        collection = self.asset.actuators
        self.native = getattr(collection, "_native_group_names", set())
        self.default = {
            name: (
                read_group_parameter(collection, name, "clamping", "max_effort")
                if name in self.native
                else collection[name].actuator_effort_limit
            ).clone()
            for name in cfg.params["actuators"]
        }

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        asset_cfg: SceneEntityCfg,
        actuators: list[str],
        strength_range: tuple[float, float] = (0.8, 1.0),
    ):
        from isaaclab.actuators.newton import write_group_parameter

        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
        collection = self.asset.actuators
        for name, default in self.default.items():
            factor = torch.empty(len(ids), default.shape[1], device=default.device).uniform_(*strength_range)
            limit = default[ids] * factor
            if name in self.native:
                write_group_parameter(collection, name, "clamping", "max_effort", limit, env_ids=ids)
            else:
                collection[name].actuator_effort_limit[ids] = limit

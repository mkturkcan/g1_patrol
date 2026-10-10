"""Observation, reward and termination terms for crowd navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster

    from .crowd import CrowdScenarioCommand


def _scenario(env: ManagerBasedRLEnv, command_name: str) -> CrowdScenarioCommand:
    return env.command_manager.get_term(command_name)


"""
Observations.
"""


def lidar_distances(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_distance: float, sectors: int | None = None
) -> torch.Tensor:
    """Lidar ranges, with misses reported as ``max_distance``.

    With ``sectors``, consecutive rays are grouped into that many equal sectors and each reports its nearest return,
    so thin obstacles (poles) that fall between the coarse sector directions are still seen. Shape (num_envs,
    sectors or num_rays).
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch
    d = torch.linalg.norm(hits - sensor.data.pos_w.torch.unsqueeze(1), dim=-1)
    d = torch.nan_to_num(d, nan=max_distance, posinf=max_distance).clamp(max=max_distance)
    if sectors is not None:
        d = d.view(d.shape[0], sectors, -1).min(dim=-1).values
    return d


"""
Rewards.
"""


def goal_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Speed towards the goal over the last step (m/s), clipped to a plausible walking speed."""
    scenario = _scenario(env, command_name)
    return ((scenario.prev_goal_dist - scenario.goal_distance()) / env.step_dt).clamp(-1.5, 1.5)


def goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    scenario = _scenario(env, command_name)
    return (scenario.goal_distance() < scenario.cfg.success_threshold).float()


def pedestrian_proximity(env: ManagerBasedRLEnv, command_name: str, comfort_distance: float) -> torch.Tensor:
    """How far (m) the robot is inside the nearest pedestrian's personal space."""
    return (comfort_distance - _scenario(env, command_name).pedestrian_clearance()).clamp(min=0.0)


def obstacle_proximity(env: ManagerBasedRLEnv, command_name: str, comfort_distance: float) -> torch.Tensor:
    """How far (m) the robot is inside the safety margin of the nearest pole or wall."""
    return (comfort_distance - _scenario(env, command_name).static_clearance()).clamp(min=0.0)


"""
Terminations.
"""


def goal_reached_termination(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    scenario = _scenario(env, command_name)
    return scenario.goal_distance() < scenario.cfg.success_threshold


def pedestrian_collision(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The robot's footprint overlaps a pedestrian."""
    scenario = _scenario(env, command_name)
    scenario.record_pedestrian_contact()  # who was moving towards whom, before the episode resets
    return scenario.pedestrian_clearance() < 0.0


def obstacle_collision(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The robot's footprint overlaps a pole or wall."""
    return _scenario(env, command_name).static_clearance() < 0.0

"""Observation, reward, termination and curriculum terms for the New York navigation task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .nyc_scenario import NycScenarioCommand


def _scenario(env: ManagerBasedRLEnv, command_name: str) -> NycScenarioCommand:
    return env.command_manager.get_term(command_name)


"""
Privileged observations (teacher).
"""


def nearby_pedestrians(env: ManagerBasedRLEnv, command_name: str, count: int = 10, max_range: float = 8.0):
    return _scenario(env, command_name).nearest_pedestrians(count, max_range)


def obstacle_sectors(env: ManagerBasedRLEnv, command_name: str, num_sectors: int = 72, max_range: float = 6.0):
    return _scenario(env, command_name).obstacle_sectors(num_sectors, max_range)


def street_state(env: ManagerBasedRLEnv, command_name: str):
    return _scenario(env, command_name).street_features()


"""
Rewards.
"""


def nyc_goal_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    s = _scenario(env, command_name)
    return ((s.prev_goal_dist - s.goal_distance()) / env.step_dt).clamp(-1.5, 1.5)


def nyc_goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    s = _scenario(env, command_name)
    return (s.goal_distance() < s.cfg.success_threshold).float()


def personal_space(env: ManagerBasedRLEnv, command_name: str, comfort_distance: float = 0.5) -> torch.Tensor:
    """How far (m) the robot is inside the nearest pedestrian's personal space."""
    return (comfort_distance - _scenario(env, command_name).pedestrian_clearance()).clamp(min=0.0)


def obstacle_margin(env: ManagerBasedRLEnv, command_name: str, margin: float = 0.15) -> torch.Tensor:
    """How far (m) the robot is inside the safety margin of the nearest structure."""
    return (margin - _scenario(env, command_name).obstacle_clearance()).clamp(min=0.0)


def pedestrian_delay(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Total walking speed (m/s) pedestrians are giving up because the robot is in their way."""
    return _scenario(env, command_name).ped_delay.sum(dim=1)


def in_street(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _scenario(env, command_name).in_street().float()


"""
Terminations.
"""


def nyc_goal_reached_termination(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    s = _scenario(env, command_name)
    return s.goal_distance() < s.cfg.success_threshold


def nyc_pedestrian_collision(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    s = _scenario(env, command_name)
    hit = s.pedestrian_clearance() < 0.0
    if hit.any():
        s.record_contacts()
    return hit


def nyc_obstacle_collision(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _scenario(env, command_name).obstacle_clearance() < 0.0


"""
Curriculum.
"""


def nyc_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    success_term: str = "goal_reached",
    failure_terms: tuple[str, ...] = ("pedestrian_collision", "obstacle_collision", "fall"),
) -> torch.Tensor:
    """Move robots to harder tiles after reaching the goal and to easier tiles after a collision or fall.

    Curriculum terms run at the start of a reset, while the termination manager still holds why the episode ended.
    """
    terrain = env.scene.terrain
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    tm = env.termination_manager
    success = tm.get_term(success_term)[env_ids]
    failure = torch.zeros_like(success)
    for name in failure_terms:
        failure |= tm.get_term(name)[env_ids]
    terrain.update_env_origins(env_ids, success, failure & ~success)
    return torch.mean(terrain.terrain_levels.float())

"""Observation, reward, termination and curriculum terms for the New York navigation task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg

    from .actions import WalkerPolicyAction
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


def nyc_goal_progress(env: ManagerBasedRLEnv, command_name: str, max_speed: float = 1.5) -> torch.Tensor:
    """Reduction of the distance to the goal per second, clipped to +-``max_speed`` (no gain from walking faster)."""
    s = _scenario(env, command_name)
    return ((s.prev_goal_dist - s.goal_distance()) / env.step_dt).clamp(-max_speed, max_speed)


def nyc_goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    s = _scenario(env, command_name)
    return (s.goal_distance() < s.cfg.success_threshold).float()


def personal_space(env: ManagerBasedRLEnv, command_name: str, comfort_distance: float = 0.5) -> torch.Tensor:
    """How far (m) the robot is inside the nearest pedestrian's personal space."""
    return (comfort_distance - _scenario(env, command_name).pedestrian_clearance()).clamp(min=0.0)


def collision_risk(env: ManagerBasedRLEnv, command_name: str, horizon: float = 2.0, margin: float = 0.1) -> torch.Tensor:
    """Predicted contacts with pedestrians within ``horizon`` seconds (see :meth:`NycScenarioCommand.collision_risk`)."""
    return _scenario(env, command_name).collision_risk(horizon, margin)


def obstacle_margin(env: ManagerBasedRLEnv, command_name: str, margin: float = 0.15) -> torch.Tensor:
    """How far (m) the robot is inside the safety margin of the nearest structure."""
    return (margin - _scenario(env, command_name).obstacle_clearance()).clamp(min=0.0)


def pedestrian_delay(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Total walking speed (m/s) pedestrians are giving up because the robot is in their way."""
    return _scenario(env, command_name).ped_delay.sum(dim=1)


def in_street(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _scenario(env, command_name).in_street().float()


def social_speed(
    env: ManagerBasedRLEnv, command_name: str, slow_speed: float = 0.3, near: float = 0.3, far: float = 1.5
) -> torch.Tensor:
    """How much (m/s) the robot walks faster than people do near others: at most ``slow_speed`` within ``near`` of a
    pedestrian (gap between bodies), rising linearly to 1 m/s at ``far``."""
    s = _scenario(env, command_name)
    allowed = slow_speed + (1.0 - slow_speed) * ((s.pedestrian_clearance() - near) / (far - near)).clamp(0.0, 1.0)
    speed = torch.linalg.norm(s.robot.data.root_lin_vel_w.torch[:, :2], dim=1)
    return (speed - allowed).clamp(min=0.0)


def obstacle_speed(
    env: ManagerBasedRLEnv, command_name: str, slow_speed: float = 0.4, near: float = 0.1, far: float = 0.4
) -> torch.Tensor:
    """How much (m/s) the robot walks faster than people do in tight spaces: at most ``slow_speed`` within ``near`` of
    a structure, rising linearly to 1 m/s at ``far``."""
    s = _scenario(env, command_name)
    allowed = slow_speed + (1.0 - slow_speed) * ((s.obstacle_clearance() - near) / (far - near)).clamp(0.0, 1.0)
    speed = torch.linalg.norm(s.robot.data.root_lin_vel_w.torch[:, :2], dim=1)
    return (speed - allowed).clamp(min=0.0)


def street_margin(env: ManagerBasedRLEnv, command_name: str, margin: float = 0.3) -> torch.Tensor:
    """How far (m) the robot is inside a safety margin along the curb of the roadway."""
    return (margin - _scenario(env, command_name).street_clearance()).clamp(min=0.0)


def _walker(env: ManagerBasedRLEnv, action_name: str) -> WalkerPolicyAction:
    return env.action_manager.get_term(action_name)


def command_lateral(env: ManagerBasedRLEnv, action_name: str = "walker") -> torch.Tensor:
    """Squared lateral velocity command: people walk forward and turn rather than sidestep."""
    return _walker(env, action_name).command[:, 1] ** 2


def command_backward(env: ManagerBasedRLEnv, action_name: str = "walker") -> torch.Tensor:
    """Squared backward velocity command."""
    return (-_walker(env, action_name).command[:, 0]).clamp(min=0.0) ** 2


def command_yaw_rate(env: ManagerBasedRLEnv, action_name: str = "walker") -> torch.Tensor:
    return _walker(env, action_name).command[:, 2] ** 2


def command_rate(env: ManagerBasedRLEnv, action_name: str = "walker") -> torch.Tensor:
    """Squared change of the applied velocity command between navigation steps."""
    w = _walker(env, action_name)
    return ((w.command - w.prev_command) ** 2).sum(dim=1)


def policy_output_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared policy output, which keeps the policy away from saturating the command squashing."""
    return (env.action_manager.action ** 2).sum(dim=1)


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


def nyc_roadway(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The robot has stepped into the roadway outside a crosswalk."""
    return _scenario(env, command_name).in_street()


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


class nyc_adaptive_curriculum(ManagerTermBase):
    """Success-driven curriculum over the (difficulty level, scenario tile) grid.

    For every tile it keeps an exponential moving average of the success rate of the episodes played on it. When an
    episode ends, the robot's next tile is drawn as follows: a scenario column uniformly at random (all scenario types
    are trained equally often), then a level within that column among the *unlocked* levels (level 0, and every level
    whose next-easier level has reached ``unlock_success``). With probability ``uniform_ratio`` the level is drawn
    uniformly among the unlocked levels (rehearsal of mastered levels), with probability ``explore_ratio`` uniformly
    among all levels (so that no level is never seen, e.g. when an intermediate level plateaus below the threshold);
    otherwise proportionally to ``s (1 - s) + floor``, which concentrates training on levels where the policy succeeds
    about half of the time.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        self.rows, self.cols = terrain.terrain_origins.shape[:2]
        levels = torch.arange(self.rows, device=env.device, dtype=torch.float)
        # optimistic prior on easy tiles, pessimistic on hard ones: training starts at the bottom of the grid
        self.success = (0.75 - 0.08 * levels)[:, None].repeat(1, self.cols).clamp(min=0.05)
        self.episodes = torch.zeros(self.rows, self.cols, device=env.device)

    def reset(self, env_ids=None):
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_term: str = "goal_reached",
        ema: float = 0.02,
        unlock_success: float = 0.5,
        uniform_ratio: float = 0.2,
        explore_ratio: float = 0.05,
        floor: float = 0.02,
    ) -> dict[str, float]:
        terrain = env.scene.terrain
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return self._stats()
        # 1. record the outcome of the episodes that just ended (not the initial reset)
        played = env.episode_length_buf[env_ids] > 0
        if played.any():
            ids = env_ids[played]
            r, c = terrain.terrain_levels[ids], terrain.terrain_types[ids]
            won = env.termination_manager.get_term(success_term)[ids].float()
            flat = r * self.cols + c
            n = torch.bincount(flat, minlength=self.rows * self.cols).float().view(self.rows, self.cols)
            wins = torch.bincount(flat, weights=won, minlength=self.rows * self.cols).view(self.rows, self.cols)
            # apply the EMA once per episode (closed form for n identical updates with the cell's mean outcome)
            keep = (1.0 - ema) ** n
            mean = torch.where(n > 0, wins / n.clamp(min=1), self.success)
            self.success = keep * self.success + (1.0 - keep) * mean
            self.episodes += n
        # 2. draw the next tile of each robot
        unlocked = torch.ones_like(self.success, dtype=torch.bool)
        unlocked[1:] = self.success[:-1] >= unlock_success
        unlocked = torch.cumprod(unlocked.int(), dim=0).bool()  # a level needs all easier levels unlocked
        weight = (self.success * (1.0 - self.success) + floor) * unlocked
        uniform = unlocked.float()
        n = len(env_ids)
        cols = torch.randint(0, self.cols, (n,), device=env.device)
        draw = torch.rand(n, device=env.device)
        probs = torch.where((draw < uniform_ratio)[:, None], uniform[:, cols].T, weight[:, cols].T)
        probs = torch.where((draw > 1.0 - explore_ratio)[:, None], torch.ones_like(probs), probs)
        rows = torch.multinomial(probs, 1).squeeze(1)
        terrain.terrain_levels[env_ids] = rows
        terrain.terrain_types[env_ids] = cols
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, cols]
        return self._stats()

    def _stats(self) -> dict[str, float]:
        unlocked = torch.ones_like(self.success, dtype=torch.bool)
        unlocked[1:] = self.success[:-1] >= self.cfg.params.get("unlock_success", 0.5)
        frontier = torch.cumprod(unlocked.int(), dim=0).sum(dim=0).float()  # unlocked levels per column
        terrain = self._env.scene.terrain
        return {
            "mean_level": float(terrain.terrain_levels.float().mean()),
            "unlocked_levels": float(frontier.mean()),
            "min_unlocked_levels": float(frontier.min()),
            "success_hard": float(self.success[(7 * self.rows) // 10 :].mean()),  # hardest 30% of the levels
        }

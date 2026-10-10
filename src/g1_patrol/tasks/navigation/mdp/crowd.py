"""Crowd scenario: a navigation goal plus a randomized layout of poles, walls and walking pedestrians.

Everything in the scenario is a kinematic rigid body, moved by writing poses (kinematic bodies on MuJoCo-Warp ignore
written velocities). :class:`CrowdScenarioCommand` owns the layout and the pedestrian state and re-plans pedestrian
velocities once per environment step with a simple social-force model; :class:`CrowdMotionAction` (an action term
with no policy inputs) integrates and writes the pedestrian poses at every physics step so their motion is smooth
and contacts with the robot see the right velocities.

All positions are 2D (x, y) relative to the environment origin unless suffixed with ``_w``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass, replace
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv


class CrowdScenarioCommand(CommandTerm):
    """Navigation goal and crowd layout. The command is the goal position in the robot's heading frame, (x, y)."""

    cfg: CrowdScenarioCommandCfg

    def __init__(self, cfg: CrowdScenarioCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.peds: RigidObjectCollection = env.scene[cfg.pedestrians_name]
        self.poles: RigidObjectCollection = env.scene[cfg.poles_name]
        self.walls: RigidObjectCollection = env.scene[cfg.walls_name]
        n, dev = self.num_envs, self.device
        self.num_peds, self.num_poles, self.num_walls = self.peds.num_bodies, self.poles.num_bodies, self.walls.num_bodies

        # goal
        self.goal_pos = torch.zeros(n, 2, device=dev)
        self.goal_pos_b = torch.zeros(n, 2, device=dev)
        self.prev_goal_dist = torch.zeros(n, device=dev)
        # static obstacles: pole centers; wall centers, yaw and half extents (half length, half thickness)
        self.pole_pos = torch.zeros(n, self.num_poles, 2, device=dev)
        self.wall_pos = torch.zeros(n, self.num_walls, 2, device=dev)
        self.wall_yaw = torch.zeros(n, self.num_walls, device=dev)
        self.wall_half = torch.tensor([cfg.wall_size[0] / 2, cfg.wall_size[1] / 2], device=dev)
        # obstacles that could not be placed are parked below the ground and ignored
        self.pole_active = torch.ones(n, self.num_poles, dtype=torch.bool, device=dev)
        self.wall_active = torch.ones(n, self.num_walls, dtype=torch.bool, device=dev)
        # pedestrians
        self.ped_pos = torch.zeros(n, self.num_peds, 2, device=dev)
        self.ped_vel = torch.zeros(n, self.num_peds, 2, device=dev)
        self.ped_goal = torch.zeros(n, self.num_peds, 2, device=dev)
        self.ped_speed = torch.zeros(n, self.num_peds, device=dev)
        self.ped_aware = torch.zeros(n, self.num_peds, dtype=torch.bool, device=dev)

        # success is reported by the goal_reached termination; the episode resets before metrics would see it
        self.metrics["min_ped_clearance"] = torch.full((n,), 10.0, device=dev)

    def __str__(self) -> str:
        return (
            f"CrowdScenarioCommand: {self.num_peds} pedestrians, {self.num_poles} poles, {self.num_walls} walls,"
            f" goal distance {self.cfg.goal_distance_range} m"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.goal_pos_b

    """
    Queries used by observations, rewards and terminations.
    """

    def robot_pos(self) -> torch.Tensor:
        return self.robot.data.root_pos_w.torch[:, :2] - self._env.scene.env_origins[:, :2]

    def goal_distance(self) -> torch.Tensor:
        return torch.linalg.norm(self.goal_pos - self.robot_pos(), dim=1)

    def pedestrian_clearance(self) -> torch.Tensor:
        """Gap between the robot and the nearest pedestrian (negative when overlapping). Shape (num_envs,)."""
        d = torch.linalg.norm(self.ped_pos - self.robot_pos().unsqueeze(1), dim=-1)
        return (d - self.cfg.pedestrian_radius - self.cfg.robot_radius).min(dim=1).values

    def record_pedestrian_contact(self):
        """Remember, per env, how the robot and its nearest pedestrian were moving (read by evaluation scripts)."""
        rel = self.ped_pos - self.robot_pos().unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1)
        nearest = dist.argmin(dim=1)
        envs = torch.arange(self.num_envs, device=self.device)
        to_ped = rel[envs, nearest] / dist[envs, nearest].unsqueeze(-1).clamp(min=1e-6)
        robot_vel = self.robot.data.root_lin_vel_w.torch[:, :2]
        self.contact_robot_approach_speed = (robot_vel * to_ped).sum(-1)
        self.contact_ped_approach_speed = -(self.ped_vel[envs, nearest] * to_ped).sum(-1)
        self.contact_ped_aware = self.ped_aware[envs, nearest]

    def static_clearance(self) -> torch.Tensor:
        """Gap between the robot and the nearest pole or wall (negative when overlapping). Shape (num_envs,)."""
        return self._static_clearance(self.robot_pos().unsqueeze(1), self.cfg.robot_radius).squeeze(1)

    """
    Implementation.
    """

    def _update_metrics(self):
        self.metrics["min_ped_clearance"] = torch.minimum(self.metrics["min_ped_clearance"], self.pedestrian_clearance())

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.metrics["min_ped_clearance"][ids] = 10.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """Lay out a new scenario around the robot's current position (called on reset)."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        cfg, n = self.cfg, len(env_ids)
        a = cfg.arena_half_size
        start = self.robot_pos()[env_ids]

        # goal at a random bearing and distance, kept inside the arena
        bearing = torch.rand(n, device=self.device) * 2 * torch.pi
        dist = torch.empty(n, device=self.device).uniform_(*cfg.goal_distance_range)
        goal = start + dist.unsqueeze(1) * torch.stack([torch.cos(bearing), torch.sin(bearing)], dim=1)
        self.goal_pos[env_ids] = goal.clamp(-a + 0.5, a - 0.5)
        keep_clear = torch.stack([start, self.goal_pos[env_ids]], dim=1)  # (n, 2, 2)

        # poles and walls anywhere in the arena, away from the start and the goal
        pos, ok = self._sample_points(n, self.num_poles, keep_clear, cfg.keep_clear_radius + cfg.pole_radius)
        self.pole_pos[env_ids], self.pole_active[env_ids] = pos, ok
        pos, ok = self._sample_points(n, self.num_walls, keep_clear, cfg.keep_clear_radius + self.wall_half[0])
        self.wall_pos[env_ids], self.wall_active[env_ids] = pos, ok
        self.wall_yaw[env_ids] = torch.rand(n, self.num_walls, device=self.device) * torch.pi
        self._write_static(env_ids)

        # pedestrians away from the robot and from static obstacles, each heading to its own goal
        self.ped_pos[env_ids] = self._sample_free(env_ids, self.num_peds, start, cfg.pedestrian_spawn_clearance)
        self.ped_goal[env_ids] = self._sample_free(env_ids, self.num_peds, None, 0.0)
        self.ped_speed[env_ids] = torch.empty(n, self.num_peds, device=self.device).uniform_(*cfg.pedestrian_speed_range)
        self.ped_aware[env_ids] = torch.rand(n, self.num_peds, device=self.device) < cfg.robot_aware_ratio
        self.ped_vel[env_ids] = 0.0
        self.write_pedestrians(env_ids)

        self.prev_goal_dist[env_ids] = torch.linalg.norm(self.goal_pos[env_ids] - start, dim=1)

    def _update_command(self):
        """Once per environment step: goal in the robot frame, and pedestrian re-planning."""
        rel = torch.zeros(self.num_envs, 3, device=self.device)
        rel[:, :2] = self.goal_pos - self.robot_pos()
        self.goal_pos_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch), rel)[:, :2]
        self.prev_goal_dist[:] = self.goal_distance()
        self._plan_pedestrians()

    def _plan_pedestrians(self):
        cfg = self.cfg
        # new goals for pedestrians that arrived
        arrived = torch.linalg.norm(self.ped_goal - self.ped_pos, dim=-1) < 0.5
        if arrived.any():
            env_ids = arrived.any(dim=1).nonzero(as_tuple=False).squeeze(1)
            new_goals = self._sample_free(env_ids, self.num_peds, None, 0.0)
            self.ped_goal[env_ids] = torch.where(arrived[env_ids].unsqueeze(-1), new_goals, self.ped_goal[env_ids])
        # desired velocity towards the goal
        to_goal = self.ped_goal - self.ped_pos
        v_des = self.ped_speed.unsqueeze(-1) * to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp(min=1e-6)

        def push(gap: torch.Tensor, away: torch.Tensor) -> torch.Tensor:
            # exponential repulsion: gap (..., k), unit vectors away from the obstacle (..., k, 2) -> (..., 2)
            return (cfg.repulsion_strength * torch.exp(-gap / cfg.repulsion_range)).unsqueeze(-1).mul(away).sum(dim=-2)

        r = cfg.pedestrian_radius
        # other pedestrians
        diff = self.ped_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)  # (n, p, p, 2)
        d = torch.linalg.norm(diff, dim=-1)
        gap = (d - 2 * r) + torch.eye(self.num_peds, device=self.device) * 1e3  # ignore self
        force = push(gap, diff / d.unsqueeze(-1).clamp(min=1e-6))
        # poles
        diff = self.ped_pos.unsqueeze(2) - self.pole_pos.unsqueeze(1)
        d = torch.linalg.norm(diff, dim=-1)
        gap = d - r - cfg.pole_radius + (~self.pole_active).unsqueeze(1) * 1e3
        force += push(gap, diff / d.unsqueeze(-1).clamp(min=1e-6))
        # walls
        closest = self._closest_on_walls(self.ped_pos)  # (n, p, w, 2)
        diff = self.ped_pos.unsqueeze(2) - closest
        d = torch.linalg.norm(diff, dim=-1)
        gap = d - r - self.wall_half[1] + (~self.wall_active).unsqueeze(1) * 1e3
        force += push(gap, diff / d.unsqueeze(-1).clamp(min=1e-6))
        # the robot, for pedestrians that pay attention to it
        diff = self.ped_pos - self.robot_pos().unsqueeze(1)
        d = torch.linalg.norm(diff, dim=-1, keepdim=True)
        gap = d.squeeze(-1) - r - cfg.robot_radius
        force += self.ped_aware.unsqueeze(-1) * push(gap.unsqueeze(-1), (diff / d.clamp(min=1e-6)).unsqueeze(-2))

        # relax towards the desired velocity, bounded speed
        step_dt = self._env.step_dt
        vel = self.ped_vel + (v_des + force - self.ped_vel) * min(step_dt / cfg.relaxation_time, 1.0)
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        self.ped_vel[:] = vel * (cfg.max_pedestrian_speed / speed.clamp(min=cfg.max_pedestrian_speed))

    def integrate_pedestrians(self, dt: float):
        """Advance pedestrians by ``dt`` and write their poses to the simulation (every physics step)."""
        a = self.cfg.arena_half_size
        self.ped_pos[:] = (self.ped_pos + self.ped_vel * dt).clamp(-a, a)
        self.write_pedestrians()

    def write_pedestrians(self, env_ids: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        pos, vel = self.ped_pos[ids], self.ped_vel[ids]
        n = pos.shape[0]
        origins = self._env.scene.env_origins[ids]
        pose = torch.zeros(n, self.num_peds, 7, device=self.device)
        pose[..., :2] = pos + origins[:, None, :2]
        pose[..., 2] = self.cfg.pedestrian_center_height
        pose[..., 6] = 1.0  # identity quaternion (x, y, z, w)
        twist = torch.zeros(n, self.num_peds, 6, device=self.device)
        twist[..., :2] = vel
        self.peds.write_body_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
        self.peds.write_body_velocity_to_sim_index(body_velocities=twist, env_ids=env_ids)

    def _write_static(self, env_ids: torch.Tensor):
        origins = self._env.scene.env_origins[env_ids]
        n = len(env_ids)
        # poles
        pose = torch.zeros(n, self.num_poles, 7, device=self.device)
        pose[..., :2] = self.pole_pos[env_ids] + origins[:, None, :2]
        pose[..., 2] = torch.where(self.pole_active[env_ids], self.cfg.pole_height / 2 + 0.01, -10.0)
        pose[..., 6] = 1.0
        self.poles.write_body_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
        # walls (yaw about z)
        pose = torch.zeros(n, self.num_walls, 7, device=self.device)
        pose[..., :2] = self.wall_pos[env_ids] + origins[:, None, :2]
        pose[..., 2] = torch.where(self.wall_active[env_ids], self.cfg.wall_size[2] / 2 + 0.01, -10.0)
        half_yaw = self.wall_yaw[env_ids] / 2
        pose[..., 5], pose[..., 6] = torch.sin(half_yaw), torch.cos(half_yaw)
        self.walls.write_body_pose_to_sim_index(body_poses=pose, env_ids=env_ids)

    """
    Geometry helpers.
    """

    def _sample_points(self, n: int, k: int, keep_clear: torch.Tensor, radius: float, tries: int = 16):
        """Sample ``k`` points per env in the arena at least ``radius`` from every point in ``keep_clear``."""
        a = self.cfg.arena_half_size - 0.5
        cand = (torch.rand(n, k, tries, 2, device=self.device) * 2 - 1) * a
        d = torch.linalg.norm(cand.unsqueeze(3) - keep_clear[:, None, None], dim=-1).min(dim=-1).values
        valid = d > radius
        first = valid.float().argmax(dim=-1)  # first valid try (0 if none)
        pos = torch.gather(cand, 2, first[..., None, None].expand(n, k, 1, 2)).squeeze(2)
        return pos, valid.any(dim=-1)

    def _sample_free(self, env_ids: torch.Tensor, k: int, avoid: torch.Tensor | None, avoid_radius: float):
        """Sample pedestrian positions clear of poles and walls (and optionally of a point per env)."""
        n, tries, a = len(env_ids), 16, self.cfg.arena_half_size - 0.5
        cand = (torch.rand(n, k, tries, 2, device=self.device) * 2 - 1) * a
        flat = cand.reshape(n, k * tries, 2)
        gap = self._static_clearance(flat, self.cfg.pedestrian_radius, env_ids).reshape(n, k, tries)
        valid = gap > 0.2
        if avoid is not None:
            valid &= torch.linalg.norm(cand - avoid[:, None, None], dim=-1) > avoid_radius
        first = valid.float().argmax(dim=-1)
        return torch.gather(cand, 2, first[..., None, None].expand(n, k, 1, 2)).squeeze(2)

    def _closest_on_walls(self, points: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Closest point on each wall's center line to each point: (n, m, 2) -> (n, m, walls, 2)."""
        ids = slice(None) if env_ids is None else env_ids
        center, yaw = self.wall_pos[ids].unsqueeze(1), self.wall_yaw[ids].unsqueeze(1)
        axis = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (n, 1, w, 2)
        t = ((points.unsqueeze(2) - center) * axis).sum(-1).clamp(-self.wall_half[0], self.wall_half[0])
        return center + t.unsqueeze(-1) * axis

    def _static_clearance(self, points: torch.Tensor, radius: float, env_ids: torch.Tensor | None = None):
        """Gap between discs of ``radius`` at ``points`` (n, m, 2) and the nearest pole or wall -> (n, m)."""
        ids = slice(None) if env_ids is None else env_ids
        d_pole = torch.linalg.norm(points.unsqueeze(2) - self.pole_pos[ids].unsqueeze(1), dim=-1)
        gap_pole = d_pole - self.cfg.pole_radius - radius + (~self.pole_active[ids]).unsqueeze(1) * 1e3
        d_wall = torch.linalg.norm(points.unsqueeze(2) - self._closest_on_walls(points, env_ids), dim=-1)
        gap_wall = d_wall - self.wall_half[1] - radius + (~self.wall_active[ids]).unsqueeze(1) * 1e3
        return torch.minimum(gap_pole.min(dim=-1).values, gap_wall.min(dim=-1).values)

    """
    Debug visualization: a marker at the goal.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        pos[:, :2] = self.goal_pos + self._env.scene.env_origins[:, :2]
        pos[:, 2] = 0.1
        self.goal_visualizer.visualize(translations=pos)


@configclass
class CrowdScenarioCommandCfg(CommandTermCfg):
    class_type: type = CrowdScenarioCommand
    asset_name: str = MISSING
    pedestrians_name: str = MISSING
    poles_name: str = MISSING
    walls_name: str = MISSING
    # a new scenario is only laid out on reset
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    arena_half_size: float = 6.0
    goal_distance_range: tuple[float, float] = (4.0, 8.0)
    success_threshold: float = 0.5
    keep_clear_radius: float = 1.0
    """Free radius around the robot's start and the goal when placing poles and walls (m)."""

    robot_radius: float = 0.25
    pedestrian_radius: float = 0.25
    pedestrian_center_height: float = 0.86
    pedestrian_spawn_clearance: float = 1.5
    pedestrian_speed_range: tuple[float, float] = (0.6, 1.4)
    max_pedestrian_speed: float = 1.8
    robot_aware_ratio: float = 0.7
    """Fraction of pedestrians that steer around the robot; the rest ignore it."""
    repulsion_strength: float = 1.5
    repulsion_range: float = 0.3
    relaxation_time: float = 0.5

    pole_radius: float = 0.1
    pole_height: float = 2.5
    wall_size: tuple[float, float, float] = (3.0, 0.2, 2.0)
    """Wall length, thickness and height (m)."""

    goal_visualizer_cfg: VisualizationMarkersCfg = replace(
        SPHERE_MARKER_CFG, prim_path="/Visuals/Command/crowd_goal"
    )

    def __post_init__(self):
        self.goal_visualizer_cfg.markers["sphere"].radius = 0.2
        self.goal_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)


class CrowdMotionAction(ActionTerm):
    """Moves the scenario's pedestrians at every physics step. Takes no policy inputs."""

    cfg: CrowdMotionActionCfg

    def __init__(self, cfg: CrowdMotionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._scenario: CrowdScenarioCommand | None = None
        self._physics_dt = env.physics_dt
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

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

    def apply_actions(self):
        if self._scenario is None:
            self._scenario = self._env.command_manager.get_term(self.cfg.scenario_name)
        self._scenario.integrate_pedestrians(self._physics_dt)


@configclass
class CrowdMotionActionCfg(ActionTermCfg):
    class_type: type = CrowdMotionAction
    scenario_name: str = MISSING

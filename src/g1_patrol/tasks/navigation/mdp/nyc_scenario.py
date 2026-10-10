"""New York scenario command: robot routes, pedestrian traffic and privileged state on :mod:`..nyc` street tiles.

Per episode, for the tile the robot is on (terrain level = difficulty, terrain type = scenario):

* the robot starts at one end of one of the tile's routes and must reach the other end;
* a difficulty-dependent number of pedestrians walk the tile's lanes. Most are timed to reach the robot's route when
  the robot does, so the crowd actually crosses and blocks its path; others stand at the tile's standing spots and
  set off later. Walkers that reach the end of a lane re-enter at its start, so traffic keeps flowing;
* pedestrians follow a simple social-force model (lanes, structures, each other, and - for those who notice it - the
  robot). Pedestrians who notice the robot slow down and step around it when it is in their way; that delay is what
  teaches the robot to give way. Distracted pedestrians (on their phone) neither slow down nor keep their distance:
  they only swerve at the last moment, so the robot has to keep out of their path.

Pedestrians are kinematic bodies moved by pose writes at every physics step (:class:`PedestrianMotionAction`).
Positions are 2D and tile-local (relative to the tile center) unless suffixed ``_w``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass, replace
from isaaclab.utils.math import quat_from_euler_xyz, yaw_quat

from ..nyc.tiles import MAX_BOXES, MAX_CIRCLES, MAX_FORBIDDEN, MAX_LANES, MAX_STANDING, OVERHEAD_Z

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv

    from ..nyc.terrain import NycTerrainImporter

PARKED, WALKING, STANDING = 0, 1, 2


def _rotate(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate 2D vectors (..., 2) by -yaw (world -> heading frame); yaw broadcasts over leading dims."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], dim=-1)


def box_distance(points: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance from points (n, m, 2) to oriented boxes (n, b, 7) and the closest box point (n, m, b, 2)."""
    center, yaw, half = boxes[:, None, :, :2], boxes[:, None, :, 2], boxes[:, None, :, 3:5]
    local = _rotate(points[:, :, None, :] - center, yaw)
    clamped = torch.maximum(torch.minimum(local, half), -half)
    c, s = torch.cos(yaw), torch.sin(yaw)
    closest = center + torch.stack([c * clamped[..., 0] - s * clamped[..., 1], s * clamped[..., 0] + c * clamped[..., 1]], -1)
    return torch.linalg.norm(local - clamped, dim=-1), closest


def ray_box_hits(origin: torch.Tensor, dirs: torch.Tensor, boxes: torch.Tensor, max_range: float) -> torch.Tensor:
    """Distance along rays (n, r, 2) from origins (n, 2) to the nearest of the oriented boxes (n, b, 7) -> (n, r)."""
    center, yaw, half = boxes[:, None, :, :2], boxes[:, None, :, 2], boxes[:, None, :, 3:5]
    o = _rotate(origin[:, None, None, :] - center, yaw)
    u = _rotate(dirs[:, :, None, :], yaw)
    inv = 1.0 / torch.where(u.abs() < 1e-6, torch.full_like(u, 1e-6), u)
    t1, t2 = (-half - o) * inv, (half - o) * inv
    t_near = torch.minimum(t1, t2).amax(dim=-1)
    t_far = torch.maximum(t1, t2).amin(dim=-1)
    hit = (t_far >= t_near) & (t_far > 0)
    t = torch.where(t_near > 0, t_near, torch.zeros_like(t_near))  # origin inside a box -> 0
    return torch.where(hit, t, torch.full_like(t, max_range)).amin(dim=-1).clamp(max=max_range)


class NycScenarioCommand(CommandTerm):
    """Goal, pedestrian traffic and privileged state. The command is the goal in the robot's heading frame (x, y)."""

    cfg: NycScenarioCommandCfg

    def __init__(self, cfg: NycScenarioCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.peds: RigidObjectCollection = env.scene[cfg.pedestrians_name]
        self.terrain: NycTerrainImporter = env.scene.terrain
        self.layout = self.terrain.layout
        n, dev = self.num_envs, self.device
        self.num_peds = self.peds.num_bodies

        # per-env copy of the current tile (tile-local coordinates)
        self.tile_center = torch.zeros(n, 2, device=dev)
        self.tile_row = torch.zeros(n, dtype=torch.long, device=dev)
        self.tile_col = torch.zeros(n, dtype=torch.long, device=dev)
        self.circles = torch.zeros(n, MAX_CIRCLES, 5, device=dev)
        self.boxes = torch.zeros(n, MAX_BOXES, 7, device=dev)
        self.forbidden = torch.zeros(n, MAX_FORBIDDEN, 4, device=dev)
        self.lanes = torch.zeros(n, MAX_LANES, 6, device=dev)
        self.standing = torch.zeros(n, MAX_STANDING, 2, device=dev)
        self.circle_valid = torch.zeros(n, MAX_CIRCLES, dtype=torch.bool, device=dev)
        self.box_valid = torch.zeros(n, MAX_BOXES, dtype=torch.bool, device=dev)
        self.forbidden_valid = torch.zeros(n, MAX_FORBIDDEN, dtype=torch.bool, device=dev)
        self.lane_valid = torch.zeros(n, MAX_LANES, dtype=torch.bool, device=dev)
        self.num_standing = torch.zeros(n, dtype=torch.long, device=dev)
        self.difficulty = torch.zeros(n, device=dev)
        # robot task
        self.start = torch.zeros(n, 2, device=dev)
        self.goal = torch.zeros(n, 2, device=dev)
        self.goal_b = torch.zeros(n, 2, device=dev)
        self.prev_goal_dist = torch.zeros(n, device=dev)
        # pedestrians
        p = self.num_peds
        self.ped_pos = torch.zeros(n, p, 2, device=dev)
        self.ped_vel = torch.zeros(n, p, 2, device=dev)
        self.ped_speed = torch.ones(n, p, device=dev)
        self.ped_aware = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.ped_mode = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.ped_lane = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.ped_offset = torch.zeros(n, p, device=dev)
        self.ped_timer = torch.zeros(n, p, device=dev)
        self.ped_delay = torch.zeros(n, p, device=dev)
        """Speed (m/s) each pedestrian is currently giving up because the robot is in its way."""

        # who was involved in the last collision (set by the collision termination, read by evaluation)
        self.contact_ped_aware = torch.zeros(n, dtype=torch.bool, device=dev)
        self.contact_robot_approach_speed = torch.zeros(n, device=dev)
        self.contact_ped_approach_speed = torch.zeros(n, device=dev)
        self.metrics["min_ped_clearance"] = torch.full((n,), 10.0, device=dev)
        self.metrics["level"] = torch.zeros(n, device=dev)

    def __str__(self) -> str:
        return f"NycScenarioCommand: {self.num_peds} pedestrians, tiles {self.layout.num_rows}x{self.layout.num_cols}"

    @property
    def command(self) -> torch.Tensor:
        return self.goal_b

    """
    Queries.
    """

    def robot_pos(self) -> torch.Tensor:
        return self.robot.data.root_pos_w.torch[:, :2] - self.tile_center

    def robot_yaw(self) -> torch.Tensor:
        q = self.robot.data.root_quat_w.torch  # (x, y, z, w)
        return torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))

    def goal_distance(self) -> torch.Tensor:
        return torch.linalg.norm(self.goal - self.robot_pos(), dim=1)

    def visible_peds(self) -> torch.Tensor:
        return self.ped_mode != PARKED

    def pedestrian_clearance(self) -> torch.Tensor:
        d = torch.linalg.norm(self.ped_pos - self.robot_pos().unsqueeze(1), dim=-1)
        gap = d - self.cfg.pedestrian_radius - self.cfg.robot_radius
        return torch.where(self.visible_peds(), gap, torch.full_like(gap, 1e3)).amin(dim=1)

    def record_contacts(self):
        """Remember, per env, whether the nearest pedestrian noticed the robot and who closed the gap faster."""
        rel = self.ped_pos - self.robot_pos().unsqueeze(1)
        d = torch.where(self.visible_peds(), torch.linalg.norm(rel, dim=-1), torch.full(rel.shape[:2], 1e3, device=self.device))
        k = d.argmin(dim=1)
        envs = torch.arange(self.num_envs, device=self.device)
        u = rel[envs, k] / d[envs, k].clamp(min=1e-3)[:, None]
        self.contact_ped_aware[:] = self.ped_aware[envs, k]
        self.contact_robot_approach_speed[:] = (self.robot.data.root_lin_vel_w.torch[:, :2] * u).sum(-1)
        self.contact_ped_approach_speed[:] = -(self.ped_vel[envs, k] * u).sum(-1)

    def obstacle_clearance(
        self, points: torch.Tensor | None = None, radius: float | None = None, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Gap between discs at points (n, m, 2) (default: the robot) and the nearest non-overhead structure.

        With ``env_ids``, ``points`` belong to those environments only.
        """
        squeeze = points is None
        points = self.robot_pos().unsqueeze(1) if points is None else points
        radius = self.cfg.robot_radius if radius is None else radius
        ids = slice(None) if env_ids is None else env_ids
        circ, circle_valid, boxes, box_valid = self.circles[ids], self.circle_valid[ids], self.boxes[ids], self.box_valid[ids]
        d_c = torch.linalg.norm(points[:, :, None] - circ[:, None, :, :2], dim=-1) - circ[:, None, :, 2]
        mask_c = circle_valid & (circ[..., 3] < OVERHEAD_Z)
        d_c = torch.where(mask_c[:, None], d_c, torch.full_like(d_c, 1e3)).amin(dim=-1)
        d_b, _ = box_distance(points, boxes)
        mask_b = box_valid & (boxes[..., 5] < OVERHEAD_Z)
        d_b = torch.where(mask_b[:, None], d_b, torch.full_like(d_b, 1e3)).amin(dim=-1)
        gap = torch.minimum(d_c, d_b) - radius
        return gap.squeeze(1) if squeeze else gap

    def in_street(self) -> torch.Tensor:
        p = self.robot_pos()
        f = self.forbidden
        inside = (p[:, None, 0] > f[..., 0]) & (p[:, None, 0] < f[..., 1]) & (p[:, None, 1] > f[..., 2]) & (p[:, None, 1] < f[..., 3])
        return (inside & self.forbidden_valid).any(dim=1)

    """
    Privileged observations (teacher).
    """

    def nearest_pedestrians(self, k: int, max_range: float) -> torch.Tensor:
        """Nearest ``k`` pedestrians: relative position, relative velocity (robot heading frame), presence.
        Shape (num_envs, 5 * k)."""
        rel = self.ped_pos - self.robot_pos().unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1)
        dist = torch.where(self.visible_peds(), dist, torch.full_like(dist, 1e3))
        order = dist.argsort(dim=1)[:, :k]
        rel = torch.gather(rel, 1, order[..., None].expand(-1, -1, 2))
        robot_vel = self.robot.data.root_lin_vel_w.torch[:, None, :2]
        vel = torch.gather(self.ped_vel, 1, order[..., None].expand(-1, -1, 2)) - robot_vel
        near = torch.gather(dist, 1, order) < max_range
        yaw = self.robot_yaw()[:, None]
        rel_b, vel_b = _rotate(rel, yaw), _rotate(vel, yaw)
        feats = torch.cat([rel_b, vel_b, near.float().unsqueeze(-1)], dim=-1) * near.unsqueeze(-1)
        return feats.reshape(self.num_envs, -1)

    def obstacle_sectors(self, num_sectors: int, max_range: float, low_height: float = 0.6) -> torch.Tensor:
        """Distance to the nearest structure surface per heading-relative sector, separately for tall and low
        (< ``low_height``, e.g. tree-pit fences, cones) obstacles. Shape (num_envs, 2 * num_sectors)."""
        n, dev = self.num_envs, self.device
        p, yaw = self.robot_pos(), self.robot_yaw()
        out = []
        for low in (False, True):
            # boxes: analytic ray casting, 2 rays per sector, min-pooled
            ang = yaw[:, None] + (torch.arange(2 * num_sectors, device=dev) + 0.5) * (math.pi / num_sectors) - math.pi
            dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
            tall_box = self.boxes[..., 6] >= low_height
            box_mask = self.box_valid & (self.boxes[..., 5] < OVERHEAD_Z) & (~tall_box if low else tall_box)
            boxes = torch.where(box_mask[..., None], self.boxes, torch.full_like(self.boxes, 1e4))  # far away
            d = ray_box_hits(p, dirs, boxes, max_range).view(n, num_sectors, 2).amin(dim=-1)
            # circles: closest surface point, scattered into its sector (exact for thin poles)
            rel = self.circles[..., :2] - p[:, None]
            dist = (torch.linalg.norm(rel, dim=-1) - self.circles[..., 2]).clamp(min=0.0)
            tall_c = self.circles[..., 4] >= low_height
            mask = self.circle_valid & (self.circles[..., 3] < OVERHEAD_Z) & (~tall_c if low else tall_c)
            dist = torch.where(mask, dist, torch.full_like(dist, max_range)).clamp(max=max_range)
            bearing = torch.atan2(rel[..., 1], rel[..., 0]) - yaw[:, None]
            sector = (((bearing + math.pi) % (2 * math.pi)) / (2 * math.pi) * num_sectors).long().clamp(0, num_sectors - 1)
            d = d.scatter_reduce(1, sector, dist, reduce="amin")
            out.append(d)
        return torch.cat(out, dim=1)

    def street_features(self) -> torch.Tensor:
        """Whether the robot is in the street, and the vector (heading frame) to the nearest street-area edge."""
        p = self.robot_pos()
        f = self.forbidden
        cx = p[:, None, 0].clamp(f[..., 0], f[..., 1])
        cy = p[:, None, 1].clamp(f[..., 2], f[..., 3])
        inside = (p[:, None, 0] > f[..., 0]) & (p[:, None, 0] < f[..., 1]) & (p[:, None, 1] > f[..., 2]) & (p[:, None, 1] < f[..., 3])
        # inside a rectangle the nearest edge is the closest of the four sides
        to_edges = torch.stack([f[..., 0] - p[:, None, 0], f[..., 1] - p[:, None, 0], f[..., 2] - p[:, None, 1], f[..., 3] - p[:, None, 1]], -1)
        k = to_edges.abs().argmin(dim=-1)
        inside_vec = torch.zeros(*k.shape, 2, device=self.device)
        inside_vec[..., 0] = torch.where(k < 2, torch.gather(to_edges, -1, k[..., None]).squeeze(-1), torch.zeros_like(cx))
        inside_vec[..., 1] = torch.where(k >= 2, torch.gather(to_edges, -1, k[..., None]).squeeze(-1), torch.zeros_like(cx))
        vec = torch.where(inside[..., None], inside_vec, torch.stack([cx, cy], -1) - p[:, None])
        dist = torch.where(self.forbidden_valid, torch.linalg.norm(vec, dim=-1), torch.full_like(cx, 1e3))
        j = dist.argmin(dim=1)
        nearest = vec[torch.arange(self.num_envs, device=self.device), j]
        nearest = torch.where((dist.amin(dim=1) < 5.0)[:, None], nearest, torch.zeros_like(nearest))
        return torch.cat([self.in_street().float().unsqueeze(-1), _rotate(nearest, self.robot_yaw())], dim=-1)

    """
    Episode setup.
    """

    def _update_metrics(self):
        self.metrics["min_ped_clearance"] = torch.minimum(self.metrics["min_ped_clearance"], self.pedestrian_clearance())
        self.metrics["level"] = self.terrain.terrain_levels.float()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.metrics["min_ped_clearance"][ids] = 10.0
        return extras

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        lay, n, dev = self.layout, len(env_ids), self.device
        r, c = self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        self.tile_row[env_ids], self.tile_col[env_ids] = r, c
        self.tile_center[env_ids] = lay.tile_centers[r, c]
        self.difficulty[env_ids] = lay.difficulty[r, c]
        cap = torch.arange
        self.circles[env_ids], self.circle_valid[env_ids] = lay.circles[r, c], cap(MAX_CIRCLES, device=dev) < lay.num_circles[r, c, None]
        self.boxes[env_ids], self.box_valid[env_ids] = lay.boxes[r, c], cap(MAX_BOXES, device=dev) < lay.num_boxes[r, c, None]
        self.forbidden[env_ids] = lay.forbidden[r, c]
        self.forbidden_valid[env_ids] = cap(MAX_FORBIDDEN, device=dev) < lay.num_forbidden[r, c, None]
        self.lanes[env_ids], self.lane_valid[env_ids] = lay.lanes[r, c], cap(MAX_LANES, device=dev) < lay.num_lanes[r, c, None]
        self.standing[env_ids], self.num_standing[env_ids] = lay.standing[r, c], lay.num_standing[r, c]

        # robot: one of the tile's routes, facing roughly towards the goal
        k = (torch.rand(n, device=dev) * lay.num_routes[r, c]).long()
        route = lay.routes[r, c, k]
        start = route[:, :2] + (torch.rand(n, 2, device=dev) - 0.5) * 0.6
        # the jitter must not put the robot against a structure (the route ends themselves are clear)
        crowded = self.obstacle_clearance(start[:, None], env_ids=env_ids)[:, 0] < 0.15
        start = torch.where(crowded[:, None], route[:, :2], start)
        self.start[env_ids], self.goal[env_ids] = start, route[:, 2:]
        heading = torch.atan2(route[:, 3] - route[:, 1], route[:, 2] - route[:, 0]) + (torch.rand(n, device=dev) - 0.5) * 1.2
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = start + self.tile_center[env_ids]
        pose[:, 2] = lay.ground_height(r, c, start[:, None])[:, 0] + self.cfg.robot_spawn_height
        zeros = torch.zeros_like(heading)
        pose[:, 3:] = quat_from_euler_xyz(zeros, zeros, heading)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self.prev_goal_dist[env_ids] = torch.linalg.norm(self.goal[env_ids] - start, dim=1)

        self._spawn_pedestrians(env_ids, route)
        self.write_pedestrians(env_ids)

    def _lane_geometry(self, lanes: torch.Tensor):
        a, b = lanes[..., 0:2], lanes[..., 2:4]
        length = torch.linalg.norm(b - a, dim=-1).clamp(min=1e-3)
        u = (b - a) / length.unsqueeze(-1)
        normal = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
        return a, u, normal, length

    def _sample_lanes(self, env_ids: torch.Tensor, count: int, route: torch.Tensor | None = None) -> torch.Tensor:
        """Lane index per pedestrian, by lane weight, favouring lanes that oppose or cross the robot's route."""
        lanes = self.lanes[env_ids]
        weight = lanes[..., 5] * self.lane_valid[env_ids]
        if route is not None:
            _, u, _, _ = self._lane_geometry(lanes)
            rdir = route[:, 2:] - route[:, :2]
            rdir = rdir / torch.linalg.norm(rdir, dim=-1, keepdim=True).clamp(min=1e-3)
            weight = weight * (1.0 + 1.5 * ((u * rdir[:, None]).sum(-1) < 0.5))
        weight = weight + 1e-6 * self.lane_valid[env_ids]
        return torch.multinomial(weight.clamp(min=1e-9), count, replacement=True)

    def _spawn_pedestrians(self, env_ids: torch.Tensor, route: torch.Tensor):
        cfg, n, p, dev = self.cfg, len(env_ids), self.num_peds, self.device
        d = self.difficulty[env_ids]
        crowd = self.layout.crowd[self.tile_row[env_ids], self.tile_col[env_ids]]
        budget = (crowd[:, 0] + (crowd[:, 1] - crowd[:, 0]) * d + torch.rand(n, device=dev) * 2 - 1).round().clamp(1, p)
        idx = torch.arange(p, device=dev)[None].expand(n, p)
        active = idx < budget[:, None]
        has_spot = (self.num_standing[env_ids] > 0)[:, None]
        standing = active & has_spot & (torch.rand(n, p, device=dev) < cfg.standing_ratio)
        walking = active & ~standing

        # attributes
        lo = cfg.speed_easy[0] + (cfg.speed_hard[0] - cfg.speed_easy[0]) * d
        hi = cfg.speed_easy[1] + (cfg.speed_hard[1] - cfg.speed_easy[1]) * d
        speed = lo[:, None] + (hi - lo)[:, None] * torch.rand(n, p, device=dev)
        slow = torch.rand(n, p, device=dev) < cfg.slow_ratio
        self.ped_speed[env_ids] = torch.where(slow, 0.4 + 0.2 * torch.rand(n, p, device=dev), speed)
        aware_prob = cfg.aware_easy + (cfg.aware_hard - cfg.aware_easy) * d
        self.ped_aware[env_ids] = torch.rand(n, p, device=dev) < aware_prob[:, None]
        lane = self._sample_lanes(env_ids, p, route)
        self.ped_lane[env_ids] = lane
        lanes = torch.gather(self.lanes[env_ids], 1, lane[..., None].expand(-1, -1, 6))
        a, u, normal, length = self._lane_geometry(lanes)
        self.ped_offset[env_ids] = (torch.rand(n, p, device=dev) * 2 - 1) * lanes[..., 4]

        # walkers: most are timed to meet the robot on its route, the rest are spread along their lanes
        meet_frac = 0.2 + 0.6 * torch.rand(n, p, device=dev)
        meet = route[:, None, :2] + meet_frac[..., None] * (route[:, None, 2:] - route[:, None, :2])
        s_meet = ((meet - a) * u).sum(-1).clamp(min=torch.zeros_like(length), max=length)
        robot_eta = torch.linalg.norm(meet - route[:, None, :2], dim=-1) / cfg.robot_nominal_speed
        s0 = s_meet - self.ped_speed[env_ids] * robot_eta
        spread = torch.rand(n, p, device=dev) > cfg.encounter_ratio
        s0 = torch.where(spread, torch.rand(n, p, device=dev) * length, s0)
        delay = torch.where(s0 < 0, -s0 / self.ped_speed[env_ids], torch.zeros_like(s0))
        s0 = s0.clamp(min=torch.zeros_like(length), max=length)
        walk_pos = a + s0[..., None] * u + self.ped_offset[env_ids][..., None] * normal

        # standing: at a random standing spot of the tile, setting off after a while
        spot = (torch.rand(n, p, device=dev) * self.num_standing[env_ids].clamp(min=1)[:, None]).long()
        spots = torch.gather(self.standing[env_ids], 1, spot[..., None].expand(-1, -1, 2))
        stand_pos = spots + (torch.rand(n, p, 2, device=dev) - 0.5) * 0.6

        mode = torch.full((n, p), PARKED, dtype=torch.long, device=dev)
        mode = torch.where(walking & (delay <= 0), torch.full_like(mode, WALKING), mode)
        mode = torch.where(standing, torch.full_like(mode, STANDING), mode)
        pos = torch.where(standing[..., None], stand_pos, walk_pos)
        timer = torch.where(standing, 5.0 + 20.0 * torch.rand(n, p, device=dev), delay)
        # nobody starts on top of the robot: those would enter later at the start of their lane instead
        near = torch.linalg.norm(pos - self.start[env_ids][:, None], dim=-1) < cfg.spawn_clearance
        mode = torch.where(near, torch.full_like(mode, PARKED), mode)
        timer = torch.where(near & (mode == PARKED), timer.clamp(min=1.0 + 3.0 * torch.rand(n, p, device=dev)), timer)
        self.ped_mode[env_ids] = mode
        self.ped_pos[env_ids] = pos
        self.ped_timer[env_ids] = torch.where(active, timer, torch.full_like(timer, 1e9))  # inactive: never appear
        self.ped_vel[env_ids] = self.ped_speed[env_ids][..., None] * u * (mode == WALKING)[..., None]
        self.ped_delay[env_ids] = 0.0

    """
    Pedestrian behaviour.
    """

    def _update_command(self):
        dt = self._env.step_dt
        p_robot, yaw = self.robot_pos(), self.robot_yaw()
        rel = torch.zeros(self.num_envs, 2, device=self.device)
        rel[:] = self.goal - p_robot
        self.goal_b[:] = _rotate(rel, yaw)
        self.prev_goal_dist[:] = self.goal_distance()
        self._plan_pedestrians(dt)

    def _plan_pedestrians(self, dt: float):
        cfg = self.cfg
        n, p, dev = self.num_envs, self.num_peds, self.device
        self.ped_timer -= dt
        lanes = torch.gather(self.lanes, 1, self.ped_lane[..., None].expand(-1, -1, 6))
        a, u, normal, length = self._lane_geometry(lanes)

        # delayed walkers appear at the start of their lane; standing people set off along a lane
        entry = a + self.ped_offset[..., None] * normal
        entry_clear = torch.linalg.norm(entry - self.robot_pos()[:, None], dim=-1) > cfg.spawn_clearance
        appear = (self.ped_mode == PARKED) & (self.ped_timer <= 0) & entry_clear
        self.ped_pos = torch.where(appear[..., None], entry, self.ped_pos)
        set_off = (self.ped_mode == STANDING) & (self.ped_timer <= 0)
        self.ped_mode = torch.where(appear | set_off, torch.full_like(self.ped_mode, WALKING), self.ped_mode)

        # walkers that reached the end of their lane re-enter at the start of a (new) lane after a short pause
        s = ((self.ped_pos - a) * u).sum(-1)
        done = (self.ped_mode == WALKING) & (s > length - 0.3)
        if done.any():
            new_lane = self._sample_lanes(torch.arange(n, device=dev), p)
            self.ped_lane = torch.where(done, new_lane, self.ped_lane)
            self.ped_timer = torch.where(done, 3.0 * torch.rand(n, p, device=dev), self.ped_timer)
            self.ped_mode = torch.where(done, torch.full_like(self.ped_mode, PARKED), self.ped_mode)
            lanes = torch.gather(self.lanes, 1, self.ped_lane[..., None].expand(-1, -1, 6))
            a, u, normal, length = self._lane_geometry(lanes)
            s = ((self.ped_pos - a) * u).sum(-1)

        # desired velocity: follow the lane (look-ahead point at the pedestrian's lateral offset)
        target = a + (s + 1.0).clamp(max=length)[..., None] * u + self.ped_offset[..., None] * normal
        to_target = target - self.ped_pos
        v_des = self.ped_speed[..., None] * to_target / torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-3)

        # give way: pedestrians who notice the robot slow down when it stands in their way
        to_robot = self.robot_pos()[:, None] - self.ped_pos
        heading = v_des / torch.linalg.norm(v_des, dim=-1, keepdim=True).clamp(min=1e-3)
        left = torch.stack([-heading[..., 1], heading[..., 0]], -1)
        ahead = (to_robot * heading).sum(-1)
        side_signed = (to_robot * left).sum(-1)  # > 0: the robot is to the pedestrian's left
        side = side_signed.abs()
        blocking = self.ped_aware & (self.ped_mode == WALKING) & (ahead > 0) & (ahead < 1.6) & (side < 0.7)
        factor = torch.where(blocking, ((ahead - 0.6) / 1.0).clamp(0.0, 1.0), torch.ones_like(ahead))
        self.ped_delay = self.ped_speed * (1.0 - factor) * blocking
        v_des = v_des * factor[..., None]
        # step around the robot: aware pedestrians early and gently, distracted ones late and abruptly
        reach = torch.where(self.ped_aware, cfg.aware_reaction_distance, cfg.distracted_reaction_distance)
        swerve = torch.where(self.ped_aware, cfg.aware_swerve_speed, cfg.distracted_swerve_speed)
        in_path = (self.ped_mode == WALKING) & (ahead > 0) & (ahead < reach) & (side < 0.7)
        closeness = (1.0 - ahead / reach).clamp(0.0, 1.0)
        away = -torch.where(side_signed >= 0, 1.0, -1.0)
        v_des = v_des + (in_path * swerve * closeness * away)[..., None] * left
        v_des = torch.where((self.ped_mode == WALKING)[..., None], v_des, torch.zeros_like(v_des))

        # social forces
        visible = self.visible_peds()

        def push(gap, away):
            return (cfg.repulsion_strength * torch.exp(-gap / cfg.repulsion_range)).unsqueeze(-1) * away

        r = cfg.pedestrian_radius
        diff = self.ped_pos[:, :, None] - self.ped_pos[:, None]
        dist = torch.linalg.norm(diff, dim=-1).clamp(min=1e-3)
        others = visible[:, None, :] & ~torch.eye(p, dtype=torch.bool, device=dev)[None]
        gap = torch.where(others, dist - 2 * r, torch.full_like(dist, 1e3))
        force = push(gap, diff / dist[..., None]).sum(dim=2)
        # structures (circles and boxes, closest surface point)
        circ = self.circles
        diff = self.ped_pos[:, :, None] - circ[:, None, :, :2]
        dist = torch.linalg.norm(diff, dim=-1).clamp(min=1e-3)
        cmask = (self.circle_valid & (circ[..., 3] < OVERHEAD_Z))[:, None]
        gap = torch.where(cmask, dist - circ[:, None, :, 2] - r, torch.full_like(dist, 1e3))
        force += push(gap, diff / dist[..., None]).sum(dim=2)
        d_box, closest = box_distance(self.ped_pos, self.boxes)
        diff = self.ped_pos[:, :, None] - closest
        bmask = (self.box_valid & (self.boxes[..., 5] < OVERHEAD_Z))[:, None]
        gap = torch.where(bmask, d_box - r, torch.full_like(d_box, 1e3))
        force += push(gap, diff / torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)).sum(dim=2)
        # the robot, for pedestrians who notice it
        d_r = torch.linalg.norm(to_robot, dim=-1).clamp(min=1e-3)
        force += self.ped_aware[..., None] * push(d_r - r - cfg.robot_radius, -to_robot / d_r[..., None])
        force = force * (self.ped_mode == WALKING)[..., None] + 0.3 * force * (self.ped_mode == STANDING)[..., None]

        vel = self.ped_vel + (v_des + force - self.ped_vel) * min(dt / cfg.relaxation_time, 1.0)
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        vel = vel * (cfg.max_pedestrian_speed / speed.clamp(min=cfg.max_pedestrian_speed))
        # aware pedestrians never walk into the robot (crowd pressure included): cap their closing speed so they keep
        # at least ``aware_min_gap`` over the next step; contact then only happens if the robot closes in itself
        u_r = to_robot / d_r[..., None]
        max_closing = ((d_r - r - cfg.robot_radius - cfg.aware_min_gap) / dt).clamp(min=0.0)
        excess = ((vel * u_r).sum(-1) - max_closing).clamp(min=0.0) * self.ped_aware
        vel = vel - excess[..., None] * u_r
        self.ped_vel = torch.where(visible[..., None], vel, torch.zeros_like(vel))

    def integrate_pedestrians(self, dt: float):
        """Advance pedestrians by ``dt`` and write their poses (every physics step)."""
        self.ped_pos = self.ped_pos + self.ped_vel * dt
        self.write_pedestrians()

    def write_pedestrians(self, env_ids: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        pos, vel, mode = self.ped_pos[ids], self.ped_vel[ids], self.ped_mode[ids]
        rows, cols = self.tile_row[ids], self.tile_col[ids]
        n, p = pos.shape[:2]
        ground = self.layout.ground_height(rows, cols, pos)
        pose = torch.zeros(n, p, 7, device=self.device)
        pose[..., :2] = pos + self.tile_center[ids][:, None]
        pose[..., 2] = torch.where(mode != PARKED, ground + self.cfg.pedestrian_center_height, torch.full_like(ground, -10.0))
        # face the walking direction (capsules are symmetric; this matters for character models)
        yaw = torch.atan2(vel[..., 1], vel[..., 0])
        pose[..., 5], pose[..., 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        twist = torch.zeros(n, p, 6, device=self.device)
        twist[..., :2] = vel * (mode != PARKED)[..., None]
        self.peds.write_body_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
        self.peds.write_body_velocity_to_sim_index(body_velocities=twist, env_ids=env_ids)

    """
    Debug visualization: goal marker.
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
        pos[:, :2] = self.goal + self.tile_center
        pos[:, 2] = self.layout.ground_height(self.tile_row, self.tile_col, self.goal[:, None])[:, 0] + 0.15
        self.goal_visualizer.visualize(translations=pos)


@configclass
class NycScenarioCommandCfg(CommandTermCfg):
    class_type: type = NycScenarioCommand
    asset_name: str = MISSING
    pedestrians_name: str = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # a new scenario only on reset

    success_threshold: float = 0.5
    robot_radius: float = 0.25
    robot_spawn_height: float = 0.8
    robot_nominal_speed: float = 0.8
    """Speed (m/s) used to time pedestrian encounters with the robot."""

    pedestrian_radius: float = 0.25
    pedestrian_center_height: float = 0.86
    speed_easy: tuple[float, float] = (0.7, 1.3)
    speed_hard: tuple[float, float] = (0.8, 1.6)
    slow_ratio: float = 0.1
    standing_ratio: float = 0.2
    encounter_ratio: float = 0.65
    """Fraction of walkers timed to reach the robot's route when the robot does."""
    aware_easy: float = 0.9
    aware_hard: float = 0.6
    """Fraction of pedestrians that notice the robot (keep their distance, slow down and step around it). The others
    are distracted: they only swerve at the last moment."""
    aware_reaction_distance: float = 2.0
    aware_swerve_speed: float = 0.4
    distracted_reaction_distance: float = 1.0
    distracted_swerve_speed: float = 0.7
    """Look-ahead (m) and peak sideways speed (m/s) with which pedestrians step around a robot in their path."""
    aware_min_gap: float = 0.1
    """Aware pedestrians never move closer than this (m) to the robot by themselves."""
    spawn_clearance: float = 1.5
    """Pedestrians never start, or enter the tile, within this distance (m) of the robot."""
    max_pedestrian_speed: float = 1.8
    repulsion_strength: float = 1.5
    repulsion_range: float = 0.3
    relaxation_time: float = 0.5

    goal_visualizer_cfg: VisualizationMarkersCfg = replace(SPHERE_MARKER_CFG, prim_path="/Visuals/Command/nyc_goal")

    def __post_init__(self):
        self.goal_visualizer_cfg.markers["sphere"].radius = 0.2
        self.goal_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)


class PedestrianMotionAction(ActionTerm):
    """Moves the scenario's pedestrians at every physics step. Takes no policy inputs."""

    cfg: PedestrianMotionActionCfg

    def __init__(self, cfg: PedestrianMotionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._scenario = None
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
        self._scenario.integrate_pedestrians(self._env.physics_dt)


@configclass
class PedestrianMotionActionCfg(ActionTermCfg):
    class_type: type = PedestrianMotionAction
    scenario_name: str = MISSING

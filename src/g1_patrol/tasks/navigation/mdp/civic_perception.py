"""What the civic students know about people, litter and places, from the G1's own sensors (plus a city map).

* :class:`tracked_people`: a person tracker on the Livox MID-360 lidar (360 degrees, 10 m) or, for the camera-only
  student, on the RealSense D435i depth stream (its field of view, about 3.6 m: the camera looks 47.6 degrees down).
  People hidden behind structures or other people are not detected; a track coasts on its last velocity for up to
  ``coast_s`` without detections. Positions and velocities are noisy. Lidar clusters tell wide bodies (wheelchairs,
  strollers) and dogs apart by size; a white cane is only recognized when the person is in the camera's view, and
  remembered. Hostility is not observed: it has to be read from how someone moves.
* :class:`detected_litter`: litter and bags detected in the D435i color image (69 x 42 degrees: the ground from about
  0.5 to 2.4 m ahead), with misses and position noise, kept in a memory in the odometry frame (which drifts), so that
  an item at the robot's toes, below the camera's view, is still known.
* :func:`known_places`: litter baskets and curbside bag piles from a city map (with localization error), the person
  who has fallen (the task's target), and the crosswalk's pedestrian signal phase from the city's signal data
  (signal phase and timing broadcasts), which is not available at every crossing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .nyc_social import DOG, CANE, WHEELCHAIR
from .nyc_trash import ON_GROUND, _rect_point

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

    from .nyc_scenario import NycScenarioCommand

D435I_POS = (0.058, 0.018, 0.430)
D435I_PITCH = 0.831
"""D435i mount on the torso link (from Unitree's g1_29dof_rev_1_0.urdf): 47.6 degrees down."""


def _scenario(env: ManagerBasedRLEnv, command_name: str) -> NycScenarioCommand:
    return env.command_manager.get_term(command_name)


def _rotate(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], dim=-1)


def camera_frame(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """World position (n, 3) and rotation matrix (n, 3, 3; columns: optical forward, left, up) of the D435i."""
    from isaaclab.utils.math import matrix_from_quat

    robot = env.scene["robot"]
    torso = robot.find_bodies("torso_link")[0][0]
    pos = robot.data.body_link_pos_w.torch[:, torso]
    rot = matrix_from_quat(robot.data.body_link_quat_w.torch[:, torso])
    offset = torch.tensor(D435I_POS, device=pos.device)
    c, s = math.cos(D435I_PITCH), math.sin(D435I_PITCH)
    pitch = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], device=pos.device)  # +pitch looks down
    return pos + (rot @ offset), rot @ pitch


def in_view(env: ManagerBasedRLEnv, points_w: torch.Tensor, h_fov: float, v_fov: float, max_range: float) -> torch.Tensor:
    """Points (n, m, 3) inside the D435i's field of view (degrees) and range."""
    pos, rot = camera_frame(env)
    rel = torch.einsum("nij,nmi->nmj", rot, points_w - pos[:, None])  # camera frame: x forward, y left, z up
    forward = rel[..., 0].clamp(min=1e-6)
    ok = rel[..., 0] > 0.05
    ok &= torch.atan2(rel[..., 1].abs(), forward) < math.radians(h_fov / 2)
    ok &= torch.atan2(rel[..., 2].abs(), forward) < math.radians(v_fov / 2)
    return ok & (torch.linalg.norm(rel, dim=-1) < max_range)


def line_of_sight(s: NycScenarioCommand, targets: torch.Tensor, skip_peds: torch.Tensor | None = None) -> torch.Tensor:
    """Tile-local points (n, m, 2) not hidden from the robot's head behind a structure (below the overhead height)
    or another (visible) pedestrian. ``skip_peds`` (n, m): the pedestrian index of each target (it does not hide
    itself), or -1."""
    from .nyc_scenario import box_distance

    n, m, _ = targets.shape
    origin = s.robot_pos()[:, None]
    d = targets - origin
    length = torch.linalg.norm(d, dim=-1).clamp(min=1e-3)
    u = d / length[..., None]
    # sample points along each sight line (every 0.25 m, up to 10 m) and test them against the structures
    steps = torch.linspace(0.08, 0.92, 12, device=targets.device)
    pts = origin[:, :, None] + d[:, :, None] * steps[None, None, :, None]  # (n, m, k, 2)
    flat = pts.reshape(n, -1, 2)
    clear = s.obstacle_clearance(flat, radius=0.0) > 0.0
    hidden = (~clear).view(n, m, -1).any(dim=-1)
    # other pedestrians closer than the target whose body covers the sight line
    rel = s.ped_pos[:, None] - origin[:, :, None]  # (n, 1, p, 2)
    along = (rel * u[:, :, None]).sum(-1)  # (n, m, p)
    across = (rel[..., 0] * u[:, :, None, 1] - rel[..., 1] * u[:, :, None, 0]).abs()
    blocker = s.visible_peds()[:, None] & (along > 0.3) & (along < length[..., None] - 0.3) & (across < s.ped_radius[:, None] * 0.9)
    if skip_peds is not None:
        blocker &= torch.arange(s.num_peds, device=targets.device)[None, None] != skip_peds[..., None]
    return ~(hidden | blocker.any(dim=-1))


class tracked_people(ManagerTermBase):
    """Nearest ``count`` tracked people (and dogs): relative position and velocity (heading frame), presence, body
    radius, wide body, dog, white cane (once recognized), the vector to the person or dog it goes with (leash,
    walking together, waiting in line; when both are tracked), and the time since the last detection. (n, 12 count)"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        s = _scenario(env, cfg.params.get("command_name", "nyc"))
        n, p, dev = env.num_envs, s.num_peds, env.device
        self.est_pos = torch.zeros(n, p, 2, device=dev)
        self.est_vel = torch.zeros(n, p, 2, device=dev)
        self.age = torch.full((n, p), 1e3, device=dev)
        self.cane_known = torch.zeros(n, p, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.age[ids] = 1e3
        self.cane_known[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "nyc",
        count: int = 8,
        sensor: str = "lidar",
        max_range: float = 10.0,
        coast_s: float = 1.0,
        pos_noise: float = 0.04,
        vel_noise: float = 0.15,
        miss: float = 0.05,
    ) -> torch.Tensor:
        s = _scenario(env, command_name)
        n, p, dev = env.num_envs, s.num_peds, env.device
        robot_p = s.robot_pos()
        rel = s.ped_pos - robot_p[:, None]
        dist = torch.linalg.norm(rel, dim=-1)
        seen = s.visible_peds() & (dist < max_range)
        if sensor == "camera":  # the depth camera sees people's legs within its field of view
            ground = s.layout.ground_height(s.tile_row, s.tile_col, s.ped_pos)
            knee = torch.cat([s.ped_pos + s.tile_center[:, None], (ground + 0.4)[..., None]], dim=-1)
            seen &= in_view(env, knee, 87.0, 58.0, max_range)
        seen &= line_of_sight(s, s.ped_pos, torch.arange(p, device=dev)[None].expand(n, p))
        seen &= torch.rand(n, p, device=dev) >= miss
        # detections update the tracks; others coast on their last velocity and are dropped after coast_s
        noise_p = torch.randn(n, p, 2, device=dev) * (pos_noise + 0.01 * dist)[..., None]
        meas_v = s.ped_vel + torch.randn(n, p, 2, device=dev) * vel_noise
        new = seen & (self.age > coast_s)
        self.est_pos = torch.where(seen[..., None], s.ped_pos + noise_p, self.est_pos + self.est_vel * env.step_dt)
        self.est_vel = torch.where(new[..., None], meas_v, torch.where(seen[..., None], 0.5 * self.est_vel + 0.5 * meas_v, self.est_vel))
        self.age = torch.where(seen, torch.zeros_like(self.age), self.age + env.step_dt)
        # a white cane is recognized in the camera's view (up close)
        ground = s.layout.ground_height(s.tile_row, s.tile_col, s.ped_pos)
        body = torch.cat([s.ped_pos + s.tile_center[:, None], (ground + 0.8)[..., None]], dim=-1)
        self.cane_known |= seen & (s.ped_role == CANE) & in_view(env, body, 69.0, 42.0, 4.0)
        self.cane_known &= s.visible_peds()
        tracked = self.age <= coast_s
        est_rel = self.est_pos - robot_p[:, None]
        d = torch.where(tracked, torch.linalg.norm(est_rel, dim=-1), torch.full_like(dist, 1e3))
        order = d.argsort(dim=1)[:, :count]

        def take(t):
            return torch.gather(t, 1, order) if t.dim() == 2 else torch.gather(t, 1, order[..., None].expand(-1, -1, t.shape[-1]))

        yaw = s.robot_yaw()[:, None]
        present = take(tracked).float()[..., None]
        robot_v = s.robot.data.root_lin_vel_w.torch[:, None, :2]
        radius = s.ped_radius + 0.03 * torch.randn(n, p, device=dev)
        other = s.ped_link.clamp(min=0)
        link_ok = (s.ped_link >= 0) & tracked & torch.gather(tracked, 1, other)
        link = torch.gather(self.est_pos, 1, other[..., None].expand(n, p, 2)) - self.est_pos
        link = torch.where(link_ok[..., None], link, torch.zeros_like(link))
        feats = torch.cat(
            [
                _rotate(take(est_rel), yaw),
                _rotate(take(self.est_vel) - robot_v, yaw),
                torch.ones_like(present),
                take(radius)[..., None],
                take((s.ped_role == WHEELCHAIR).float())[..., None],
                take((s.ped_role == DOG).float())[..., None],
                take(self.cane_known.float())[..., None],
                _rotate(take(link), yaw),
                take(self.age).clamp(max=coast_s)[..., None],
            ],
            dim=-1,
        )
        return (feats * present).reshape(n, -1)


class detected_litter(ManagerTermBase):
    """Nearest ``count`` remembered litter items and bags: relative position (heading frame, from the drifting
    odometry), height of the grasp point above the robot's feet, bag, time since last seen. (n, 5 count)"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        s = _scenario(env, cfg.params.get("command_name", "nyc"))
        n, k, dev = env.num_envs, max(s.num_trash, 1), env.device
        self.mem = torch.zeros(n, k, 2, device=dev)  # odometry frame (world + drift)
        self.age = torch.full((n, k), 1e3, device=dev)
        self.drift = torch.zeros(n, 3, device=dev)  # odometry error: x, y, yaw

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.age[ids] = 1e3
        self.drift[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "nyc",
        count: int = 3,
        litter_range: float = 4.0,
        bag_range: float = 6.0,
        memory_s: float = 30.0,
        drift_pos: float = 0.005,
        drift_yaw: float = 0.002,
    ) -> torch.Tensor:
        s = _scenario(env, command_name)
        n, dev = env.num_envs, env.device
        out = torch.zeros(n, count, 5, device=dev)
        if s.num_trash == 0:
            return out.reshape(n, -1)
        k = s.num_trash
        self.drift += torch.randn(n, 3, device=dev) * torch.tensor([drift_pos, drift_pos, drift_yaw], device=dev)
        robot_p, yaw = s.robot_pos(), s.robot_yaw()

        def to_odom(p):  # true tile-local -> estimated (drifting) frame
            rel = p - robot_p[:, None]
            c, s_ = torch.cos(self.drift[:, 2])[:, None], torch.sin(self.drift[:, 2])[:, None]
            return robot_p[:, None] + self.drift[:, None, :2] + torch.stack([c * rel[..., 0] - s_ * rel[..., 1], s_ * rel[..., 0] + c * rel[..., 1]], -1)

        pos = s.trash_positions()
        on = s.trash_state == ON_GROUND
        dist = torch.linalg.norm(pos[..., :2] - robot_p[:, None], dim=-1)
        rng = torch.where(s.is_bag[None], bag_range, litter_range)
        seen = on & in_view(env, s.trash.data.body_link_pos_w.torch, 69.0, 42.0, 10.0) & (dist < rng)
        seen &= line_of_sight(s, pos[..., :2])
        p_det = torch.where(s.is_bag[None], 0.99, (0.97 - 0.08 * dist).clamp(0.5, 0.97))
        seen &= torch.rand(n, k, device=dev) < p_det
        meas = pos[..., :2] + torch.randn(n, k, 2, device=dev) * (0.02 + 0.015 * dist)[..., None]
        self.mem = torch.where(seen[..., None], to_odom(meas), self.mem)
        self.age = torch.where(seen, torch.zeros_like(self.age), self.age + env.step_dt)
        # items that are no longer lying there (picked up, put away, gone into the street) are forgotten
        self.age = torch.where(on, self.age, torch.full_like(self.age, 1e3))
        known = self.age < memory_s
        est_robot = robot_p + self.drift[:, :2]
        rel = self.mem - est_robot[:, None]
        d = torch.where(known, torch.linalg.norm(rel, dim=-1), torch.full_like(dist, 1e3))
        order = d.argsort(dim=1)[:, : min(count, k)]
        present = torch.gather(known, 1, order).float()[..., None]
        rel_b = _rotate(torch.gather(rel, 1, order[..., None].expand(-1, -1, 2)), (yaw + self.drift[:, 2])[:, None])
        feet = s.layout.ground_height(s.tile_row, s.tile_col, robot_p[:, None])
        h = torch.gather(pos[..., 2], 1, order) + s.trash_grasp_height[order] - feet
        bag = s.is_bag[order].float()[..., None]
        age = torch.gather(self.age, 1, order).clamp(max=memory_s)[..., None] / memory_s
        out[:, : order.shape[1]] = torch.cat([rel_b, h[..., None], bag, age], dim=-1) * present
        return out.reshape(n, -1)


def known_places(
    env: ManagerBasedRLEnv, command_name: str = "nyc", map_noise: float = 0.1, signal_coverage: float = 0.7
) -> torch.Tensor:
    """From the city map and data: the nearest two litter baskets (relative position, presence), the nearest bag
    zone's closest point (relative position, presence), the person who has fallen (relative position of the body,
    presence) and the crosswalk's signal (phase one-hot, time left / 30 s, available) where the signal broadcasts.
    (n, 17)"""
    s = _scenario(env, command_name)
    n, dev = env.num_envs, env.device
    baskets = s.nearest_baskets(2)
    piles = s.nearest_piles(1)
    jitter = torch.randn(n, 3, 2, device=dev) * map_noise
    baskets = baskets.view(n, 2, 3)
    baskets = torch.cat([baskets[..., :2] + jitter[:, :2] * baskets[..., 2:], baskets[..., 2:]], dim=-1).view(n, -1)
    piles = torch.cat([piles[:, :2] + jitter[:, 2] * piles[:, 2:], piles[:, 2:]], dim=-1)
    fallen = s.fallen_features()[:, [0, 1, 3]] if s.social else torch.zeros(n, 3, device=dev)
    signal = s.signal_features()
    # broadcasts are available at a fixed share of the crossings (decided per tile)
    covered = ((s.tile_row * 7 + s.tile_col * 13) % 10).float() < 10 * signal_coverage
    signal = signal * covered[:, None]
    return torch.cat([baskets, piles, fallen, signal], dim=1)

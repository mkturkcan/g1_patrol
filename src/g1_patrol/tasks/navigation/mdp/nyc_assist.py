"""Civic tasks: someone who has fallen, and pedestrian signals at crosswalks.

Fallen person. In some episodes a person lies on the sidewalk ahead of the robot. The robot's task is then to help:
walk up to the person (its goal becomes a point beside them), stop at a respectful distance (0.5-1.8 m from their body,
facing them) and raise the alarm (the interaction trigger; on the robot this calls for help). It must not touch them.
For the other pedestrians and the collision checks the person is a low obstacle.

Signals. Crosswalks have pedestrian signals: WALK, flashing DON'T WALK, steady DON'T WALK. Starting to cross on
anything but WALK is a violation (as stepping into the roadway); someone already crossing finishes the crossing. While
crossing is not allowed, the crosswalk counts as roadway for the robot's street observations, so it waits at the curb.
Most pedestrians wait at the curb too; a few cross against the signal.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz

from ..nyc.tiles import MAX_BOXES, MAX_FORBIDDEN
from .nyc_social import FALLEN, NO_INCIDENT

WALK, FLASH, DONT_WALK = 0, 1, 2
FALLEN_HALF = (0.85, 0.22)
"""Half length and half width (m) of a person lying on the ground."""
FALLEN_BOX = MAX_BOXES - 1
"""Box slot of the per-episode obstacles that holds the fallen person (unused by the tiles)."""
SIGNAL_SLOT = MAX_FORBIDDEN - 1
"""Street-area slot that holds the crosswalk while crossing is not allowed (unused by the tiles)."""


class AssistMixin:
    """Fallen person and crosswalk signals; mixed into :class:`NycScenarioCommand`."""

    def _init_assist(self, cfg, env):
        n, dev = self.num_envs, self.device
        self.fallen = env.scene[cfg.fallen_name] if cfg.fallen_name is not None else None
        self.fallen_pos = torch.zeros(n, 2, device=dev)
        self.fallen_yaw = torch.zeros(n, device=dev)
        self.assisted = torch.zeros(n, dtype=torch.bool, device=dev)
        """The robot has raised the alarm next to the fallen person (set by the hands controller)."""
        self.alert_event = torch.zeros(n, device=dev)
        self.metrics["fallen"] = torch.zeros(n, device=dev)
        self.metrics["assisted"] = torch.zeros(n, device=dev)
        self._fallen_episode = torch.zeros(n, device=dev)
        # signals
        self.signal_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.signal_phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.signal_timer = torch.zeros(n, device=dev)
        self.crossing_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.signal_violation = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_crossing = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ped_jaywalk = torch.zeros(n, self.num_peds, dtype=torch.bool, device=dev)

    @property
    def has_fallen(self) -> torch.Tensor:
        return self.incident == FALLEN

    # -- episode setup ------------------------------------------------------------------------------------------------

    def _assist_metrics(self, env_ids):
        self.metrics["fallen"][env_ids] = self._fallen_episode[env_ids]
        self.metrics["assisted"][env_ids] = self.assisted[env_ids].float()

    def _spawn_assist(self, env_ids: torch.Tensor, route: torch.Tensor):
        cfg, n, dev = self.cfg, len(env_ids), self.device
        e = env_ids
        self.assisted[e] = False
        self._fallen_episode[e] = 0.0
        if cfg.signals:
            self._spawn_signals(e)
        if not self.social or self.fallen is None:
            return
        self.box_valid[e, FALLEN_BOX] = False
        want = (self.incident[e] == NO_INCIDENT) & (torch.rand(n, device=dev) < self._ranged(cfg.fallen_ratio, self.difficulty[e]))
        placed = torch.zeros(n, dtype=torch.bool, device=dev)
        if want.any():
            placed, spot, yaw, approach = self._place_fallen(e, route)
            placed &= want
            rows = placed.nonzero().flatten()
            ee = e[rows]
            self.incident[ee] = FALLEN
            self.incident_phase[ee] = 3  # DONE: nothing more happens; the robot must act
            self.fallen_pos[ee], self.fallen_yaw[ee] = spot[rows], yaw[rows]
            self.goal[ee] = approach[rows]
            self.prev_goal_dist[ee] = torch.linalg.norm(self.goal[ee] - self.start[ee], dim=1)
            box = torch.zeros(len(rows), 7, device=dev)
            box[:, :2], box[:, 2] = spot[rows], yaw[rows]
            box[:, 3], box[:, 4], box[:, 5], box[:, 6] = FALLEN_HALF[0], FALLEN_HALF[1], 0.0, 0.35
            self.boxes[ee, FALLEN_BOX] = box
            self.box_valid[ee, FALLEN_BOX] = True
            self._fallen_episode[ee] = 1.0
        # pose of the person's body (a lying capsule), or parked below the ground
        pose = torch.zeros(n, 7, device=dev)
        r, c = self.tile_row[e], self.tile_col[e]
        ground = self.layout.ground_height(r, c, self.fallen_pos[e][:, None])[:, 0]
        pose[:, :2] = self.fallen_pos[e] + self.tile_center[e]
        pose[:, 2] = torch.where(placed, ground + FALLEN_HALF[1], torch.full_like(ground, -10.0))
        zeros = torch.zeros(n, device=dev)
        pose[:, 3:] = quat_from_euler_xyz(zeros, zeros, self.fallen_yaw[e])
        self.fallen.write_root_pose_to_sim_index(root_pose=pose, env_ids=e)
        self.fallen.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=e)

    def _place_fallen(self, e: torch.Tensor, route: torch.Tensor):
        """A spot 35-80% along the route, up to 1.2 m from it, with room around the body, on the sidewalk; and the
        point 1.2 m from the body on the robot's side, which becomes the robot's goal."""
        n, dev, m = len(e), self.device, 24
        start, goal = route[:, :2], route[:, 2:]
        along = goal - start
        u = along / torch.linalg.norm(along, dim=1, keepdim=True).clamp(min=1e-3)
        normal = torch.stack([-u[:, 1], u[:, 0]], dim=1)
        t = 0.35 + 0.45 * torch.rand(n, m, device=dev)
        side = (2 * torch.rand(n, m, device=dev) - 1) * 1.2
        spot = start[:, None] + t[..., None] * along[:, None] + side[..., None] * normal[:, None]
        yaw = 2 * math.pi * torch.rand(n, m, device=dev)
        # footprint corners and centre must clear the structures; nothing in the street
        c, s = torch.cos(yaw), torch.sin(yaw)
        hx, hy = FALLEN_HALF
        corners = torch.stack([spot + torch.stack([c * a * hx - s * b * hy, s * a * hx + c * b * hy], -1)
                               for a in (-1, 1) for b in (-1, 1)] + [spot], dim=2)  # (n, m, 5, 2)
        flat = corners.reshape(n, m * 5, 2)
        ok = (self.obstacle_clearance(flat, radius=0.35, env_ids=e) > 0.0).reshape(n, m, 5).all(dim=2)
        ok &= ~self._street_mask(flat, e).reshape(n, m, 5).any(dim=2)
        to_start = self.start[e][:, None] - spot
        dist_start = torch.linalg.norm(to_start, dim=-1)
        ok &= dist_start > 3.0
        # the goal: 1 m from the body's edge, on the side the robot comes from
        u = to_start / dist_start[..., None].clamp(min=1e-3)
        ub = torch.stack([c * u[..., 0] + s * u[..., 1], -s * u[..., 0] + c * u[..., 1]], dim=-1)  # body frame
        edge = torch.minimum(hx / ub[..., 0].abs().clamp(min=1e-3), hy / ub[..., 1].abs().clamp(min=1e-3))
        approach = spot + (edge + 1.0)[..., None] * u
        ok &= self.obstacle_clearance(approach, radius=self.cfg.robot_radius + 0.1, env_ids=e) > 0.0
        ok &= ~self._street_mask(approach, e)
        r, col = self.tile_row[e], self.tile_col[e]
        ground = self.layout.ground_height(r, col, spot)
        ok &= (ground - self.layout.ground_height(r, col, self.start[e][:, None])).abs() < 0.3
        k = ok.float().argmax(dim=1)
        rows = torch.arange(n, device=dev)
        return ok.any(dim=1), spot[rows, k], yaw[rows, k], approach[rows, k]

    def fallen_distance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance (m) from the robot's center to the fallen person's body, and the bearing of the body's center
        relative to the robot's heading (rad)."""
        from .nyc_scenario import box_distance

        p = self.robot_pos()
        box = self.boxes[:, FALLEN_BOX][:, None]
        d, _ = box_distance(p[:, None], box)
        rel = self.fallen_pos - p
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - self.robot_yaw()
        bearing = torch.remainder(bearing + math.pi, 2 * math.pi) - math.pi
        return d[:, 0, 0], bearing

    def can_alert(self) -> torch.Tensor:
        d, bearing = self.fallen_distance()
        lo, hi = self.cfg.alert_distance
        return self.has_fallen & ~self.assisted & (d > lo) & (d < hi) & (bearing.abs() < self.cfg.alert_angle)

    def fallen_features(self) -> torch.Tensor:
        """Fallen person: position of the body's center (heading frame), distance to the body, presence. (n, 4)."""
        d, _ = self.fallen_distance()
        rel = self._to_heading((self.fallen_pos - self.robot_pos())[:, None])[:, 0]
        on = self.has_fallen & ~self.assisted
        return torch.cat([rel, d[:, None].clamp(max=10.0), torch.ones_like(d)[:, None]], dim=1) * on[:, None]

    # -- signals --------------------------------------------------------------------------------------------------------

    def _phase_duration(self, phase: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        lo = torch.tensor([cfg.walk_time[0], cfg.flash_time[0], cfg.dont_walk_time[0]], device=self.device)[phase]
        hi = torch.tensor([cfg.walk_time[1], cfg.flash_time[1], cfg.dont_walk_time[1]], device=self.device)[phase]
        return lo + (hi - lo) * torch.rand_like(lo)

    def _spawn_signals(self, e: torch.Tensor):
        n = len(e)
        valid = self.crossing_valid[e, 0]
        self.signal_valid[e] = valid
        phase = torch.randint(0, 3, (n,), device=self.device)
        self.signal_phase[e] = phase
        self.signal_timer[e] = self._phase_duration(phase) * torch.rand(n, device=self.device)
        self.crossing_ok[e] = False
        self._in_crossing[e] = False
        self.signal_violation[e] = False
        self.ped_jaywalk[e] = torch.rand(n, self.num_peds, device=self.device) < self._ranged(self.cfg.jaywalk_ratio, self.difficulty[e])[:, None]
        self.forbidden[e, SIGNAL_SLOT] = self.crossings[e, 0]
        self.forbidden_valid[e, SIGNAL_SLOT] = False

    def update_signals(self, dt: float):
        if not self.cfg.signals:
            return
        self.signal_timer -= dt
        due = self.signal_valid & (self.signal_timer <= 0.0)
        if due.any():
            nxt = (self.signal_phase[due] + 1) % 3
            self.signal_phase[due] = nxt
            self.signal_timer[due] = self._phase_duration(nxt)
        p = self.robot_pos()
        rect = self.crossings[:, 0]
        inside = self.signal_valid & (p[:, 0] > rect[:, 0]) & (p[:, 0] < rect[:, 1]) & (p[:, 1] > rect[:, 2]) & (p[:, 1] < rect[:, 3])
        entered = inside & ~self._in_crossing
        self.crossing_ok = torch.where(entered, self.signal_phase == WALK, self.crossing_ok & inside)
        self._in_crossing = inside
        self.signal_violation = inside & ~self.crossing_ok
        # while crossing is not allowed (and the robot is not already crossing), the crosswalk is roadway
        self.forbidden_valid[:, SIGNAL_SLOT] = self.signal_valid & (self.signal_phase != WALK) & ~self.crossing_ok

    def signal_hold(self, v_des: torch.Tensor, path_dir: torch.Tensor) -> torch.Tensor:
        """Pedestrians about to step onto a crosswalk without WALK wait at the curb (except the few who jaywalk)."""
        if not self.cfg.signals or not self.signal_valid.any():
            return v_des
        rect = self.crossings[:, 0][:, None]
        def inside(q):
            return (q[..., 0] > rect[..., 0]) & (q[..., 0] < rect[..., 1]) & (q[..., 1] > rect[..., 2]) & (q[..., 1] < rect[..., 3])
        ahead = self.ped_pos + 0.7 * path_dir
        stop = self.signal_valid[:, None] & (self.signal_phase != WALK)[:, None] & ~inside(self.ped_pos) & inside(ahead)
        stop &= ~self.ped_jaywalk & (self.ped_role != 4)  # dogs follow their owners
        return torch.where(stop[..., None], torch.zeros_like(v_des), v_des)

    def signal_features(self) -> torch.Tensor:
        """Pedestrian signal of the tile's crosswalk: WALK, flashing, DON'T WALK (one-hot), time left in the phase
        (/ 30 s), crosswalk present. (n, 5)."""
        one_hot = torch.nn.functional.one_hot(self.signal_phase, 3).float()
        f = torch.cat([one_hot, (self.signal_timer / 30.0)[:, None], torch.ones_like(self.signal_timer)[:, None]], dim=1)
        return f * self.signal_valid[:, None]

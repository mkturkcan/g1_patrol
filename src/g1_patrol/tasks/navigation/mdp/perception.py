"""Observation terms for the G1's own head sensors (students): a Livox MID-360 lidar and a RealSense D435i camera.

* :class:`lidar_elevation_map` bins lidar returns into a robot-centred, heading-aligned height map (the usual
  "elevation mapping" input for legged robots): curbs, potholes, low fences, poles and people all show up as heights.
  The last ``num_frames`` maps are stacked so moving people are visible.
* :class:`depth_frames` stacks the last ``num_frames`` depth images with sensor noise, missing pixels and occasional
  occlusions (a hand over the camera, glare), so the camera-only policy learns to cope with a degraded view.
* :class:`accumulated_height_map` integrates the returns of either sensor over the last seconds into a robot-centred
  height map (odometry is taken as known), so that structures stay in the map after they leave the field of view.

Both return 2D observation groups (channels, height, width) for a CNN encoder.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from g1_patrol.heightmap import HeightMap, drop_returns, rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import RayCaster, RayCasterCamera


class lidar_elevation_map(ManagerTermBase):
    """Stacked height maps (num_frames, cells, cells) from a lidar's returns, in the robot's heading frame.

    Cell values are the highest return in the cell relative to the robot's ground level, clipped to ``z_range``;
    cells without returns are ``unknown_value``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.sensor_cfg: SceneEntityCfg = p["sensor_cfg"]
        self.cells = int(round(p.get("size", 8.0) / p.get("resolution", 0.2)))
        self.resolution = p.get("resolution", 0.2)
        self.num_frames = p.get("num_frames", 2)
        self.frames = torch.full(
            (env.num_envs, self.num_frames, self.cells, self.cells), p.get("unknown_value", -1.0), device=env.device
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.frames[ids] = self.cfg.params.get("unknown_value", -1.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        size: float = 8.0,
        resolution: float = 0.2,
        num_frames: int = 2,
        z_range: tuple[float, float] = (-0.5, 2.0),
        unknown_value: float = -1.0,
        robot_ground_offset: float = 0.76,
    ) -> torch.Tensor:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        robot = env.scene["robot"]
        hits = sensor.data.ray_hits_w.torch  # (n, rays, 3)
        n = hits.shape[0]
        root = robot.data.root_pos_w.torch
        q = robot.data.root_quat_w.torch
        yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        rel = hits - root[:, None]
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        x = c * rel[..., 0] + s * rel[..., 1]
        y = -s * rel[..., 0] + c * rel[..., 1]
        z = (rel[..., 2] + robot_ground_offset).clamp(*z_range)
        half = self.cells * self.resolution / 2
        i = ((x + half) / self.resolution).floor().long()
        j = ((y + half) / self.resolution).floor().long()
        valid = torch.isfinite(rel).all(dim=-1) & (i >= 0) & (i < self.cells) & (j >= 0) & (j < self.cells)
        flat = (i * self.cells + j).clamp(0, self.cells * self.cells - 1)
        z = torch.where(valid, z, torch.full_like(z, z_range[0] - 1.0))
        grid = torch.full((n, self.cells * self.cells), z_range[0] - 1.0, device=hits.device)
        grid = grid.scatter_reduce(1, flat, z, reduce="amax")
        grid = torch.where(grid < z_range[0], torch.full_like(grid, unknown_value), grid)
        self.frames = torch.cat([grid.view(n, 1, self.cells, self.cells), self.frames[:, :-1]], dim=1)
        return self.frames.clone()


class depth_frames(ManagerTermBase):
    """Stacked, degraded depth images (num_frames, height, width), normalized to [0, 1] by ``max_depth``."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        sensor: RayCasterCamera = env.scene.sensors[p["sensor_cfg"].name]
        h, w = sensor.image_shape
        self.num_frames = p.get("num_frames", 3)
        self.frames = torch.ones((env.num_envs, self.num_frames, h, w), device=env.device)
        # per-env occlusion: a blob over part of the image for a few seconds at a time
        self.occluded_until = torch.zeros(env.num_envs, device=env.device)
        self.occlusion_box = torch.zeros(env.num_envs, 4, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.frames[ids] = 1.0
        self.occluded_until[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        max_depth: float = 6.0,
        num_frames: int = 3,
        noise_std: float = 0.02,
        dropout: float = 0.05,
        occlusion_rate: float = 0.02,
    ) -> torch.Tensor:
        sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
        depth = sensor.data.output["distance_to_image_plane"].torch[..., 0]  # (n, h, w)
        depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth).clamp(0.0, max_depth) / max_depth
        n, h, w = depth.shape
        if noise_std > 0:  # stereo depth noise grows with distance
            depth = depth + torch.randn_like(depth) * noise_std * (0.5 + depth)
        if dropout > 0:  # missing pixels (no stereo match) read as zero, like the real camera
            depth = torch.where(torch.rand_like(depth) < dropout, torch.zeros_like(depth), depth)
        if occlusion_rate > 0:
            now = env.episode_length_buf.float() * env.step_dt
            start = (torch.rand(n, device=depth.device) < occlusion_rate * env.step_dt) & (now >= self.occluded_until)
            if start.any():
                self.occluded_until[start] = now[start] + 0.5 + 2.0 * torch.rand(int(start.sum()), device=depth.device)
                bh, bw = (torch.rand(2, int(start.sum()), device=depth.device) * 0.5 + 0.2) * torch.tensor([[h], [w]], device=depth.device)
                top = (torch.rand(int(start.sum()), device=depth.device) * (h - bh)).long()
                left = (torch.rand(int(start.sum()), device=depth.device) * (w - bw)).long()
                self.occlusion_box[start] = torch.stack([top, top + bh.long(), left, left + bw.long()], dim=1)
            rows = torch.arange(h, device=depth.device)[None, :, None]
            cols = torch.arange(w, device=depth.device)[None, None, :]
            b = self.occlusion_box
            inside = (rows >= b[:, 0, None, None]) & (rows < b[:, 1, None, None]) & (cols >= b[:, 2, None, None]) & (cols < b[:, 3, None, None])
            active = (now < self.occluded_until)[:, None, None]
            depth = torch.where(inside & active, torch.zeros_like(depth), depth)
        depth = depth.clamp(0.0, 1.0)
        self.frames = torch.cat([depth[:, None], self.frames[:, :-1]], dim=1)
        return self.frames.clone()


class accumulated_height_map(ManagerTermBase):
    """Height map accumulated over the last ``memory_s`` seconds, in the robot's heading frame (3, cells, cells).

    The returns of each scan are binned into a world-aligned grid that scrolls with the robot; the robot's motion between
    scans is taken as known (on the robot it comes from lidar-inertial or visual-inertial odometry). Observed cells take
    the highest return of the latest scan; a cell is forgotten ``memory_s`` seconds after it was last observed. The
    output is resampled into the robot's heading frame (rows: forward, columns: left) with three channels: the
    accumulated map, the current scan and the previous scan (motion-compensated, so moving pedestrians are visible).
    Heights are relative to the robot's ground level and clipped to ``z_range``; unknown cells are ``unknown_value``.

    ``sensor_cfg`` may name a lidar (:class:`RayCaster`) or a depth camera (:class:`RayCasterCamera`); ``noise_std``
    (relative range noise, growing with distance) and ``dropout`` (missing returns) degrade the returns as in
    :class:`depth_frames`. The robot's pose estimate drifts from its true pose as a random walk of ``drift_pos`` (m)
    and ``drift_yaw`` (rad) per step, so that older parts of the map are misaligned as with real odometry.

    As on the robot, returns can be filtered before they are binned: ``crop_box`` (center and half sizes in the base
    frame) removes returns from the robot's own body, ``drop_above`` removes returns higher than this above the
    ground (ceilings, overhangs the robot passes under).

    ``extra_sensors`` fuses further sensors into the same map (e.g. depth cameras next to a lidar): one dict per sensor
    with ``sensor_cfg`` and optionally ``max_range``, ``min_range``, ``noise_std``, ``dropout`` and
    ``episode_dropout``, the probability that the sensor delivers nothing for a whole episode (a camera that failed).

    ``floor_dropout`` (min, max): per episode, a share drawn from this range of the returns from the floor farther than
    ``floor_dropout_from`` m from the robot is lost -- polished floors reflect the lidar and the depth cameras' light
    away at shallow angles.
    """

    UNKNOWN = HeightMap.UNKNOWN

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.map = HeightMap(env.num_envs, p.get("size", 8.0), p.get("resolution", 0.2), env.device)
        self.pose_error = torch.zeros(env.num_envs, 3, device=env.device)  # odometry drift: x, y, yaw
        self.sensor_off = torch.zeros(env.num_envs, len(p.get("extra_sensors", ())), dtype=torch.bool, device=env.device)
        self.floor_loss = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.map.reset(env_ids)

    def _points(self, env, sensor_cfg, max_range, noise_std, dropout, min_range: float = 0.0) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        # lidars expose their hits in the data buffer, ray-cast cameras on the sensor
        hits = (sensor.data.ray_hits_w if hasattr(sensor.data, "ray_hits_w") else sensor.ray_hits_w).torch
        origin = sensor.data.pos_w.torch[:, None]
        hits = torch.nan_to_num(hits, nan=float("inf"))
        ray = hits - origin
        dist = torch.linalg.norm(ray, dim=-1, keepdim=True)
        ok = torch.isfinite(dist) & (dist < max_range)
        if min_range > 0:
            ok = ok & (dist >= min_range)
        if noise_std > 0:
            scale = 1.0 + torch.randn_like(dist) * noise_std * (0.5 + dist / max_range)
            hits = origin + torch.where(ok, ray * scale, ray)
        if dropout > 0:
            ok = ok & (torch.rand_like(dist) >= dropout)
        return torch.where(ok, hits, torch.full_like(hits, float("inf")))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        size: float = 8.0,
        resolution: float = 0.2,
        memory_s: float = 3.0,
        max_range: float = 10.0,
        noise_std: float = 0.0,
        dropout: float = 0.0,
        z_range: tuple[float, float] = (-0.5, 2.0),
        unknown_value: float = -1.0,
        robot_ground_offset: float = 0.76,
        drift_pos: float = 0.0,
        drift_yaw: float = 0.0,
        crop_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        drop_above: float | None = None,
        extra_sensors: tuple[dict, ...] = (),
        floor_dropout: tuple[float, float] = (0.0, 0.0),
        floor_dropout_from: float = 2.0,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        true_root = robot.data.root_pos_w.torch
        q = robot.data.root_quat_w.torch
        true_yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        n = true_root.shape[0]
        # estimated pose: the true pose plus a drifting odometry error (zero at the start of an episode)
        fresh = self.map.fresh
        self.pose_error[fresh] = 0.0
        if drift_pos > 0 or drift_yaw > 0:
            step = torch.randn(n, 3, device=true_root.device) * torch.tensor([drift_pos, drift_pos, drift_yaw], device=true_root.device)
            self.pose_error += torch.where(fresh[:, None], torch.zeros_like(step), step)
        root = true_root.clone()
        root[:, :2] += self.pose_error[:, :2]
        yaw = true_yaw + self.pose_error[:, 2]

        # the scan, filtered as on the robot and placed with the estimated pose
        pts = self._points(env, sensor_cfg, max_range, noise_std, dropout)
        if extra_sensors:
            if fresh.any():  # which sensors fail for the episode
                rates = torch.tensor([e.get("episode_dropout", 0.0) for e in extra_sensors], device=pts.device)
                self.sensor_off[fresh] = torch.rand(int(fresh.sum()), len(extra_sensors), device=pts.device) < rates
            parts = [pts]
            for k, e in enumerate(extra_sensors):
                more = self._points(env, e["sensor_cfg"], e.get("max_range", max_range), e.get("noise_std", 0.0),
                                    e.get("dropout", 0.0), e.get("min_range", 0.0))
                parts.append(torch.where(self.sensor_off[:, k, None, None], torch.full_like(more, float("inf")), more))
            pts = torch.cat(parts, dim=1)
        if floor_dropout[1] > 0:  # polished floor: far floor returns go missing
            if fresh.any():
                lo, hi = floor_dropout
                self.floor_loss[fresh] = lo + (hi - lo) * torch.rand(int(fresh.sum()), device=pts.device)
            ground_z = true_root[:, None, 2] - robot_ground_offset
            far = torch.linalg.norm(torch.nan_to_num(pts[..., :2] - true_root[:, None, :2], posinf=1e3), dim=-1) > floor_dropout_from
            floor = (pts[..., 2] - ground_z).abs() < 0.05
            lost = floor & far & (torch.rand_like(pts[..., 0]) < self.floor_loss[:, None])
            pts = torch.where(lost[..., None], torch.full_like(pts, float("inf")), pts)
        if crop_box is not None or drop_above is not None:
            local = rotate_inverse(q[:, None].expand(-1, pts.shape[1], -1), pts - true_root[:, None]) if crop_box is not None else None
            pts = drop_returns(pts, local, true_root[:, 2] - robot_ground_offset, crop_box, drop_above)
        rel = pts - true_root[:, None]
        c, s = torch.cos(self.pose_error[:, 2])[:, None], torch.sin(self.pose_error[:, 2])[:, None]
        pts = torch.stack([root[:, None, 0] + c * rel[..., 0] - s * rel[..., 1],
                           root[:, None, 1] + s * rel[..., 0] + c * rel[..., 1], pts[..., 2]], dim=-1)
        ground = root[:, 2] - robot_ground_offset
        return self.map.update(pts, root, yaw, ground, env.step_dt, memory_s, z_range, unknown_value)

"""Observation terms for the G1's own head sensors (students): a Livox MID-360 lidar and a RealSense D435i camera.

* :class:`lidar_elevation_map` bins lidar returns into a robot-centred, heading-aligned height map (the usual
  "elevation mapping" input for legged robots): curbs, potholes, low fences, poles and people all show up as heights.
  The last ``num_frames`` maps are stacked so moving people are visible.
* :class:`depth_frames` stacks the last ``num_frames`` depth images with sensor noise, missing pixels and occasional
  occlusions (a hand over the camera, glare), so the camera-only policy learns to cope with a degraded view.

Both return 2D observation groups (channels, height, width) for a CNN encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

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

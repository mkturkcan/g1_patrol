"""Robot-centred height map accumulated from 3D returns -- shared by simulation and the robot.

Pure torch (no simulator imports), so that the observation the policy is trained on and the one computed on the
robot's computer from the lidar driver and odometry come from the same code: the simulation's
:class:`g1_patrol.tasks.navigation.mdp.accumulated_height_map` and the B2-W deployment node both feed their returns
to :class:`HeightMap`.

Returns are binned into a world-aligned grid that scrolls with the robot; each observed cell takes the highest return
of the latest scan and is forgotten ``memory_s`` seconds after it was last observed. The output is resampled into the
robot's heading frame (rows: forward, columns: left, centred on the robot) with three channels: the accumulated map,
the current scan and the previous scan (motion-compensated, so that moving people stand out). Heights are relative to
the ground under the robot and clipped to ``z_range``; unknown cells read ``unknown_value``.

For telling what moved (:func:`g1_patrol.shield.moving_returns`), the map also keeps how long ago each cell last held
something taller than ``TALL`` (:attr:`HeightMap.tall_age_out`, heading frame, before the latest scan): the edges of
static things flicker between floor and tall from scan to scan as the sensors' rays fall differently, whereas a
person walking into open floor enters cells that were floor for a while.
"""

from __future__ import annotations

import math

import torch


def rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors (..., 3) by the inverse of quaternions (..., 4) in (x, y, z, w) order."""
    q, w = quat[..., :3], quat[..., 3:4]
    t = 2.0 * torch.cross(q, vec, dim=-1)
    return vec - w * t + torch.cross(q, t, dim=-1)


def drop_returns(points: torch.Tensor, local: torch.Tensor, ground: torch.Tensor,
                 crop_box: tuple | None = None, drop_above: float | None = None) -> torch.Tensor:
    """Replace by +inf the returns (n, k, 3) that are the robot itself -- inside ``crop_box`` ((center, half sizes) in
    the base frame; ``local`` are the returns in the base frame) -- or higher than ``drop_above`` above ``ground``
    (n,), i.e. what the robot passes under."""
    if crop_box is not None:
        center = torch.tensor(crop_box[0], device=points.device)
        half = torch.tensor(crop_box[1], device=points.device)
        inside = ((torch.nan_to_num(local, posinf=1e3, neginf=-1e3) - center).abs() <= half).all(dim=-1, keepdim=True)
        points = torch.where(inside, torch.full_like(points, float("inf")), points)
    if drop_above is not None:
        high = points[..., 2:3] - ground[:, None, None] > drop_above
        points = torch.where(high, torch.full_like(points, float("inf")), points)
    return points


class HeightMap:
    """Accumulated height maps of ``num`` robots, ``size`` x ``size`` m at ``resolution`` (see the module)."""

    UNKNOWN = -1.0e3
    TALL = 0.5
    """A return this high (m) above the ground makes its cell tall (for :attr:`tall_age_out`)."""

    def __init__(self, num: int, size: float = 8.0, resolution: float = 0.2, device: str | torch.device = "cpu"):
        self.res = resolution
        self.cells = int(round(size / resolution))
        self.map_cells = int(math.ceil(self.cells * math.sqrt(2))) + 2  # the rotated output window fits inside
        n, m = num, self.map_cells
        self.height = torch.full((n, m, m), self.UNKNOWN, device=device)
        self.age = torch.full((n, m, m), 1.0e3, device=device)
        self.prev_scan = torch.full((n, m, m), self.UNKNOWN, device=device)
        self.tall_age = torch.full((n, m, m), 1.0e3, device=device)
        self.tall_age_out: torch.Tensor | None = None
        """Seconds (n, cells, cells; heading frame) since each cell last held something tall, before the latest scan."""
        self.origin = torch.zeros(n, 2, dtype=torch.long, device=device)  # world cell index of map cell (0, 0)
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)
        # output cell centres in the robot frame
        c = (torch.arange(self.cells, device=device, dtype=torch.float) + 0.5) * self.res - self.cells * self.res / 2
        self.local = torch.stack(torch.meshgrid(c, c, indexing="ij"), dim=-1).view(-1, 2)  # (cells^2, 2): x, y

    def reset(self, ids=None):
        """Start the maps of ``ids`` (default: all) empty at their next update."""
        self.fresh[slice(None) if ids is None else ids] = True

    def _shift(self, grid: torch.Tensor, d: torch.Tensor, fill: float) -> torch.Tensor:
        """grid[i, j] <- grid[i + d0, j + d1] per robot; cells shifted in from outside are ``fill``."""
        n, m = grid.shape[0], self.map_cells
        ar = torch.arange(m, device=grid.device)
        si = ar[None, :, None] + d[:, 0, None, None]
        sj = ar[None, None, :] + d[:, 1, None, None]
        valid = (si >= 0) & (si < m) & (sj >= 0) & (sj < m)
        flat = (si.clamp(0, m - 1) * m + sj.clamp(0, m - 1)).view(n, -1)
        out = grid.view(n, -1).gather(1, flat).view(n, m, m)
        return torch.where(valid, out, torch.full_like(out, fill))

    def update(
        self,
        points: torch.Tensor,
        root: torch.Tensor,
        yaw: torch.Tensor,
        ground: torch.Tensor,
        dt: float,
        memory_s: float = 3.0,
        z_range: tuple[float, float] = (-0.5, 2.0),
        unknown_value: float = -1.0,
    ) -> torch.Tensor:
        """Add a scan and return the maps (n, 3, cells, cells).

        Args:
            points: Returns in the world (odometry) frame (n, k, 3); +inf or NaN where there is none.
            root: Robot positions (n, 3) and ``yaw`` headings (n,) in the same frame (estimates, on the robot).
            ground: Height (n,) of the ground under each robot in that frame.
            dt: Time since the previous update (s).
        """
        n, m, res = root.shape[0], self.map_cells, self.res
        # scroll the map with the robot (maps that were reset start empty)
        origin = torch.floor(root[:, :2] / res).long() - m // 2
        if self.fresh.any():
            f = self.fresh
            self.height[f], self.age[f], self.prev_scan[f] = self.UNKNOWN, 1.0e3, self.UNKNOWN
            self.tall_age[f] = 1.0e3
            self.origin[f] = origin[f]
            self.fresh[:] = False
        d = origin - self.origin
        if (d != 0).any():
            self.height = self._shift(self.height, d, self.UNKNOWN)
            self.age = self._shift(self.age, d, 1.0e3)
            self.prev_scan = self._shift(self.prev_scan, d, self.UNKNOWN)
            self.tall_age = self._shift(self.tall_age, d, 1.0e3)
            self.origin = origin

        # bin the scan into the grid: highest return per cell
        ci = torch.floor(points[..., 0] / res) - origin[:, 0, None]
        cj = torch.floor(points[..., 1] / res) - origin[:, 1, None]
        valid = torch.isfinite(points).all(dim=-1) & (ci >= 0) & (ci < m) & (cj >= 0) & (cj < m)
        flat = torch.where(valid, ci * m + cj, torch.zeros_like(ci)).long()
        z = torch.where(valid, points[..., 2], torch.full_like(ci, self.UNKNOWN))
        scan = torch.full((n, m * m), self.UNKNOWN, device=root.device).scatter_reduce(1, flat, z, reduce="amax")
        scan = scan.view(n, m, m)
        seen = scan > self.UNKNOWN / 2

        # when each cell last held something tall (the time before this scan is kept for the output)
        tall_before = self.tall_age
        tall = seen & (scan - ground[:, None, None] > self.TALL)
        self.tall_age = torch.where(tall, torch.zeros_like(tall_before), tall_before + dt)

        # latest observation wins; unobserved cells age and are forgotten after memory_s
        self.height = torch.where(seen, scan, self.height)
        self.age = torch.where(seen, torch.zeros_like(self.age), self.age + dt)
        self.height = torch.where(self.age > memory_s, torch.full_like(self.height, self.UNKNOWN), self.height)

        # resample the three grids into the robot's heading frame
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        lx, ly = self.local[None, :, 0], self.local[None, :, 1]
        wx = root[:, 0, None] + c * lx - s * ly
        wy = root[:, 1, None] + s * lx + c * ly
        gi = (torch.floor(wx / res).long() - origin[:, 0, None]).clamp(0, m - 1)
        gj = (torch.floor(wy / res).long() - origin[:, 1, None]).clamp(0, m - 1)
        idx = gi * m + gj
        out = []
        for grid in (self.height, scan, self.prev_scan):
            v = grid.view(n, -1).gather(1, idx)
            known = v > self.UNKNOWN / 2
            v = torch.where(known, (v - ground[:, None]).clamp(*z_range), torch.full_like(v, unknown_value))
            out.append(v.view(n, self.cells, self.cells))
        self.tall_age_out = tall_before.view(n, -1).gather(1, idx).view(n, self.cells, self.cells)
        self.prev_scan = scan
        return torch.stack(out, dim=1)


def map_statistics(maps, unknown: float, resolution: float) -> dict:
    """Statistics of height maps (T, 3, H, W) (numpy or torch), to compare drives on the robot with simulation: the
    share of known cells (all, within 1.5 m), of the current scan, of obstacle cells (> 0.3 m), and the median height of
    the floor 1-3 m around the robot (about 0 if the base height and the lidar mount are right)."""
    import numpy as np

    maps = np.asarray(maps)
    acc = maps[:, 0]
    known = acc > unknown + 1e-6
    h, w = acc.shape[1:]
    yy, xx = np.meshgrid((np.arange(h) + 0.5 - h / 2) * resolution, (np.arange(w) + 0.5 - w / 2) * resolution, indexing="ij")
    dist = np.hypot(xx, yy)
    near, ring = dist < 1.5, (dist > 1.0) & (dist < 3.0)
    floor = acc[:, ring][known[:, ring] & (np.abs(acc[:, ring]) < 0.15)]
    return {
        "known": float(known.mean()),
        "known_near_1.5m": float(known[:, near].mean()),
        "current_scan_known": float((maps[:, 1] > unknown + 1e-6).mean()),
        "floor_height_median": float(np.median(floor)) if floor.size else float("nan"),
        "obstacle_cells_over_0.3m": float((acc > 0.3).mean()),
    }

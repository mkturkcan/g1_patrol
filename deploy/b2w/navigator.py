"""Robot-side navigation for the Unitree B2-W: observations from the sensors, odometry and route, the policy, the shield.

Everything the robot computes before and after the policy lives here, without ROS, so that the ROS 2 node
(:mod:`ros2_node`), the replay of recorded drives (``scripts/b2w/replay.py``) and the tests all run the same code. The
height map is :class:`g1_patrol.heightmap.HeightMap`, the same implementation the policy was trained with; the
settings (map size, crop box, mount, command ranges) come from ``deploy.json`` written by ``scripts/b2w/export.py``.

The policy follows a route, as it did in training: a global planner's path (e.g. Nav2's ``/plan``), or a straight line
to a goal in open space. It is given the points of the route 1 and 2.5 m (``deploy.json``) ahead of the robot's place
along it (:mod:`g1_patrol.route`, the same code as in training).

Frames: the odometry frame is the world (x, y, z; yaw about z), the base frame is base_link (x forward, y left, z up),
a depth camera's optical frame has z along its axis, x right and y down. Poses and routes must be in the same frame
(odometry, or a map frame with localization).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from g1_patrol.floor import FloorEstimator
from g1_patrol.heightmap import HeightMap, drop_returns
from g1_patrol.route import path_project, polyline, waypoints
from g1_patrol.shield import ShieldCfg, ShieldGuard, map_points, moving_returns


@dataclass
class Pose:
    """Pose of base_link in the odometry frame."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def rotation(self) -> np.ndarray:
        """Rotation matrix base -> odometry frame (roll, pitch, yaw about x, y, z)."""
        cr, sr, cp, sp, cy, sy = (math.cos(self.roll), math.sin(self.roll), math.cos(self.pitch), math.sin(self.pitch),
                                  math.cos(self.yaw), math.sin(self.yaw))
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])


def _euler_rotation(rpy) -> np.ndarray:
    return Pose(0.0, 0.0, 0.0, *rpy).rotation()


class OnnxPolicy:
    """The exported student (``policy.onnx``: inputs ``obs`` (1, 14) and the map (1, 3, H, W); output ``actions``)."""

    def __init__(self, path: str | Path, providers: list[str] | None = None, threads: int = 2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # idle onnxruntime threads must not spin: they would take the CPU from the height map between steps
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), options, providers=providers or ort.get_available_providers())
        self.inputs = [i.name for i in self.session.get_inputs()]

    def __call__(self, obs: np.ndarray, height_map: np.ndarray) -> np.ndarray:
        feed = {self.inputs[0]: obs.astype(np.float32)[None], self.inputs[1]: height_map.astype(np.float32)[None]}
        return self.session.run(None, feed)[0][0]


class Navigator:
    """Turns lidar scans, odometry and a route into velocity commands for the B2-W's controller.

    Call :meth:`set_path` (a planner's path; again whenever it replans) or :meth:`set_goal` (a straight line), then for
    each lidar scan :meth:`step` with the scan and the current pose; it returns the command (forward, lateral, yaw
    rate) and a dict of diagnostics. One policy step per scan, as in training (lidar and policy at 10 Hz).
    """

    def __init__(self, config: dict | str | Path, policy=None, device: str = "cpu", max_speed: float | None = None,
                 guard: bool = True, shield_cfg: ShieldCfg | None = None, threads: int = 2, crawl: bool = True):
        # the tensors are small: more threads only cost (with one thread per core on a large machine, 60x slower)
        torch.set_num_threads(threads)
        self.cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
        m = self.cfg["map"]
        self.device = device
        self.map = HeightMap(1, m["size"], m["resolution"], device)
        # the floor below the base, measured in every scan (the standing height is not assumed; deploy.json's is a prior)
        self.floor = FloorEstimator(prior=-float(m.get("base_height", 0.65)))
        self.floor_z = self.floor.prior
        self.policy = policy
        ranges = np.asarray(self.cfg["command"]["ranges"], dtype=np.float64)
        self.mid, self.half = ranges.mean(axis=1), (ranges[:, 1] - ranges[:, 0]) / 2
        mount = self.cfg["lidar"]["mount"]
        self.mount_pos = np.asarray(mount["pos"], dtype=np.float64)
        self.mount_rot = _euler_rotation(mount.get("rpy", (0.0, 0.0, 0.0)))
        self.cameras = {name: (np.asarray(c["pos"], dtype=np.float64), _euler_rotation(c["rpy"]), c)
                        for name, c in self.cfg.get("depth_cameras", {}).items()}
        self.max_speed = max_speed
        # the cap is also an input of the policy, trained with caps in speed_limit_range (none: the highest)
        lo, hi = self.cfg["speed_limit_range"]
        self.speed_limit = float(np.clip(hi if max_speed is None else max_speed, lo, hi))
        self.guard = guard
        crawling = {} if crawl else {"near_moving_gap": 0.0}
        self.shield_cfg = shield_cfg or ShieldCfg(**self.cfg.get("shield", {}), **crawling)
        # the shield with what the robot does around it: alternatives to a blocked command, backing off when stuck,
        # crawling near what moved unless crawl=False (the same code as the evaluation in simulation)
        self.shield_guard = ShieldGuard(1, self.shield_cfg)
        self.goal: np.ndarray | None = None
        self.route: tuple[torch.Tensor, torch.Tensor] | None = None
        self.route_s: torch.Tensor | None = None
        self.route_version = 0
        self.lookahead = torch.tensor(self.cfg["route_waypoints"], dtype=torch.float32)
        self.last_action = np.zeros(3)
        self.last_stamp: float | None = None
        self.last_scan_base: np.ndarray | None = None
        self.rng = np.random.default_rng(0)

    def reset(self):
        self.map.reset()
        self.floor.reset()
        self.shield_guard.reset()
        self.last_action[:] = 0.0
        self.last_stamp = None

    def set_goal(self, x: float, y: float):
        """Goal reached in a straight line from where the robot is at the next step (open spaces; elsewhere a planner's
        path, :meth:`set_path`). The map and the last action are kept."""
        self.goal = np.array([x, y], dtype=np.float64)
        self.route = None

    def clear(self):
        """No goal: the robot stands (until :meth:`set_goal` or :meth:`set_path`)."""
        self.goal, self.route = None, None

    def set_path(self, points):
        """Route (k, 2) to follow, e.g. a global planner's path (again whenever it replans); its last point is the goal.
        The robot's place along a new route is the closest point of the whole route, then searched near the last."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            raise ValueError("empty path")
        pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3]]  # without repeated points
        if len(pts) == 1:
            pts = np.repeat(pts, 2, axis=0)
        self.goal = pts[-1].copy()
        self.route = polyline(torch.as_tensor(pts, dtype=torch.float32)[None])
        self.route_s = None
        self.route_version += 1

    # -- observations -------------------------------------------------------------------------------------------------

    def scan_to_base(self, points_lidar: np.ndarray) -> np.ndarray:
        """Lidar returns (k, 3) in the lidar frame -> base frame."""
        return points_lidar @ self.mount_rot.T + self.mount_pos

    def depth_to_base(self, name: str, points_optical: np.ndarray) -> np.ndarray:
        """Depth-camera returns (k, 3) in its optical frame -> base frame, within the camera's range used in training."""
        pos, rot, c = self.cameras[name]
        pts = np.asarray(points_optical, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r >= c["min_range"]) & (r < c["max_range"])]
        return pts @ rot.T + pos

    def update_map(self, points_lidar: np.ndarray, pose: Pose, dt: float, depth: dict | None = None) -> np.ndarray:
        """Add a scan taken at ``pose`` (and the latest depth frames, ``depth``: camera name -> returns in its optical
        frame) and return the map (3, H, W) in the robot's heading frame."""
        m = self.cfg["map"]
        base = self.scan_to_base(np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3))
        base = base[np.isfinite(base).all(axis=1)]
        rng = np.linalg.norm(base - self.mount_pos, axis=1)
        base = base[rng < m["max_range"]]
        limit = self.cfg["lidar"].get("max_points")
        if limit and len(base) > limit:  # thin to the simulated lidar's density (the map's max per cell depends on it)
            base = base[self.rng.choice(len(base), limit, replace=False)]
        for name, pts in (depth or {}).items():
            if name in self.cameras and pts is not None:
                base = np.concatenate([base, self.depth_to_base(name, pts)])
        self.last_scan_base = base
        world = base @ pose.rotation().T + np.array([pose.x, pose.y, pose.z])
        # the floor below the base (gravity-aligned): the map's heights are taken from it
        self.floor_z = self.floor.update(world - np.array([pose.x, pose.y, pose.z]))
        dev = self.device
        points = torch.as_tensor(world, dtype=torch.float32, device=dev)[None]
        local = torch.as_tensor(base, dtype=torch.float32, device=dev)[None]
        ground = torch.tensor([pose.z + self.floor_z], dtype=torch.float32, device=dev)
        points = drop_returns(points, local, ground, tuple(map(tuple, m["crop_box"])), m["drop_above"])
        root = torch.tensor([[pose.x, pose.y, pose.z]], dtype=torch.float32, device=dev)
        yaw = torch.tensor([pose.yaw], dtype=torch.float32, device=dev)
        out = self.map.update(points, root, yaw, ground, dt, m["memory_s"], tuple(m["z_range"]), m["unknown_value"])
        return out[0].cpu().numpy()

    def follow(self, pose: Pose) -> np.ndarray:
        """Update the robot's place along its route; the route's points ahead of it in the heading frame (2W,)."""
        if self.goal is None:
            raise RuntimeError("no goal")
        x = torch.tensor([[pose.x, pose.y]], dtype=torch.float32)
        if self.route is None:  # straight to the goal
            self.set_path([[pose.x, pose.y], self.goal])
        pts, cum = self.route
        self.route_s = path_project(pts, cum, x, self.route_s)
        yaw = torch.tensor([pose.yaw], dtype=torch.float32)
        return waypoints(pts, cum, self.route_s, x, yaw, self.lookahead)[0].flatten().numpy().astype(np.float64)

    def proprio(self, pose: Pose, velocity_b: np.ndarray) -> np.ndarray:
        """Route waypoints (heading frame), velocity (body frame: forward, lateral, yaw rate), projected gravity, last
        action, speed limit."""
        gravity_b = pose.rotation().T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([self.follow(pose), np.asarray(velocity_b, dtype=np.float64), gravity_b, self.last_action,
                               [self.speed_limit]])

    # -- commands -----------------------------------------------------------------------------------------------------

    def to_command(self, action: np.ndarray) -> np.ndarray:
        """Policy output -> velocity command, as in training (tanh into the command ranges)."""
        return self.mid + self.half * np.tanh(action)

    def shielded(self, command: np.ndarray, velocity_b: np.ndarray, height_map: np.ndarray | None,
                 person: np.ndarray | None = None, dt: float = 0.1) -> tuple[np.ndarray, dict]:
        """The safety shield (:mod:`g1_patrol.shield`) on the latest returns and the map's remembered obstacles: never
        into anything seen, slowly towards people. ``person``: which of the latest returns a detector marks."""
        if not self.guard or self.last_scan_base is None:
            return command, {}
        pts = torch.as_tensor(self.last_scan_base, dtype=torch.float32)[None]
        mask = None if person is None else torch.as_tensor(person, dtype=torch.bool)[None]
        moving = None
        if height_map is not None:
            res = self.cfg["map"]["resolution"]
            hm = torch.as_tensor(height_map, dtype=torch.float32)[None]
            tall_age = None if self.map.tall_age_out is None else self.map.tall_age_out[:1].cpu()
            moving = moving_returns(pts, hm, res, self.floor_z, tall_age=tall_age,
                                    recent=self.shield_cfg.still_for, min_cells=self.shield_cfg.moving_cells)
            cells = map_points(hm[:, 0], res, self.floor_z)  # obstacles that left the sensors' view
            pts = torch.cat([pts, cells], dim=1)
            moving = torch.cat([moving, torch.zeros(1, cells.shape[1], dtype=torch.bool)], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.zeros(1, cells.shape[1], dtype=torch.bool)], dim=1)
        vel = torch.as_tensor(velocity_b, dtype=torch.float32)[None]
        safe, info = self.shield_guard(pts, torch.as_tensor(command, dtype=torch.float32)[None], dt, vel, moving, mask,
                                       self.floor_z)
        return safe[0].numpy().astype(np.float64), {k: float(v[0]) for k, v in info.items()}

    def step(self, points_lidar: np.ndarray, pose: Pose, velocity_b: np.ndarray, stamp: float,
             depth: dict | None = None, person: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """One policy step for a new scan (and the latest depth frames): the command (forward m/s, lateral m/s, yaw
        rate rad/s) and diagnostics."""
        dt = 0.1 if self.last_stamp is None else float(np.clip(stamp - self.last_stamp, 0.02, 0.5))
        self.last_stamp = stamp
        height_map = self.update_map(points_lidar, pose, dt, depth)
        info = {"map": height_map, "dt": dt}
        if self.goal is None:
            return np.zeros(3), {**info, "state": "no goal"}
        distance = float(np.hypot(*(self.goal - np.array([pose.x, pose.y]))))
        info["goal_distance"] = distance
        if distance < self.cfg["success_threshold"]:
            self.last_action[:] = 0.0
            return np.zeros(3), {**info, "state": "reached"}
        obs = self.proprio(pose, velocity_b)
        action = np.asarray(self.policy(obs, height_map), dtype=np.float64)
        self.last_action = action.copy()
        command = self.to_command(action)
        if self.max_speed is not None:
            speed = float(np.hypot(command[0], command[1]))
            if speed > self.max_speed:
                command[:2] *= self.max_speed / speed
        info["policy_command"] = command.copy()
        command, report = self.shielded(command, velocity_b, height_map, person, dt)
        state = ("emergency stop" if report.get("emergency") else "backing off" if report.get("backing_off")
                 else "crawling (people near)" if report.get("crawling") else "shield"
                 if report.get("scale", 1.0) < 0.999 else "driving")
        return command, {**info, "state": state, "shield": report, "obs": obs, "action": action}


def depth_image_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int,
                       scale: float = 0.001) -> np.ndarray:
    """Returns (k, 3) in the optical frame from a depth image (z per pixel, ``scale`` m per unit; 0 = no return),
    sampled on a ``width`` x ``height`` grid of pixels -- the rays of the simulated camera."""
    h, w = depth.shape
    us = np.linspace(0, w - 1, width).round().astype(int)
    vs = np.linspace(0, h - 1, height).round().astype(int)
    u, v = np.meshgrid(us, vs)
    z = depth[v, u].astype(np.float64) * scale
    ok = z > 0
    return np.stack([(u[ok] - cx) * z[ok] / fx, (v[ok] - cy) * z[ok] / fy, z[ok]], axis=1)

"""Robot-side navigation for the Unitree B2-W: observations from the lidar, odometry and goal, the policy, a guard.

Everything the robot computes before and after the policy lives here, without ROS, so that the ROS 2 node
(:mod:`ros2_node`), the replay of recorded drives (``scripts/b2w/replay.py``) and the tests all run the same code. The
height map is :class:`g1_patrol.heightmap.HeightMap`, the same implementation the policy was trained with; the
settings (map size, crop box, mount, command ranges) come from ``deploy.json`` written by ``scripts/b2w/export.py``.

Frames: the odometry frame is the world (x, y, z; yaw about z), the base frame is base_link (x forward, y left, z up).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from g1_patrol.heightmap import HeightMap, drop_returns


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
    """The exported student (``policy.onnx``: inputs ``obs`` (1, 11) and the map (1, 3, H, W); output ``actions``)."""

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
    """Turns lidar scans, odometry and a goal into velocity commands for the B2-W's controller.

    Call :meth:`set_goal`, then for each lidar scan :meth:`step` with the scan and the current odometry; it returns
    the command (forward, lateral, yaw rate) and a dict of diagnostics. One policy step per scan, as in training
    (lidar and policy at 10 Hz).
    """

    def __init__(self, config: dict | str | Path, policy=None, device: str = "cpu", max_speed: float | None = None,
                 guard: bool = True):
        self.cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
        m = self.cfg["map"]
        self.device = device
        self.map = HeightMap(1, m["size"], m["resolution"], device)
        self.policy = policy
        ranges = np.asarray(self.cfg["command"]["ranges"], dtype=np.float64)
        self.mid, self.half = ranges.mean(axis=1), (ranges[:, 1] - ranges[:, 0]) / 2
        mount = self.cfg["lidar"]["mount"]
        self.mount_pos = np.asarray(mount["pos"], dtype=np.float64)
        self.mount_rot = _euler_rotation(mount.get("rpy", (0.0, 0.0, 0.0)))
        self.max_speed = max_speed
        self.guard = guard
        self.goal: np.ndarray | None = None
        self.last_action = np.zeros(3)
        self.last_stamp: float | None = None
        self.last_scan_base: np.ndarray | None = None
        self.rng = np.random.default_rng(0)

    def reset(self):
        self.map.reset()
        self.last_action[:] = 0.0
        self.last_stamp = None

    def set_goal(self, x: float, y: float):
        """Goal in the odometry frame; the map and the last action are kept."""
        self.goal = np.array([x, y], dtype=np.float64)

    # -- observations -------------------------------------------------------------------------------------------------

    def scan_to_base(self, points_lidar: np.ndarray) -> np.ndarray:
        """Lidar returns (k, 3) in the lidar frame -> base frame."""
        return points_lidar @ self.mount_rot.T + self.mount_pos

    def update_map(self, points_lidar: np.ndarray, pose: Pose, dt: float) -> np.ndarray:
        """Add a scan taken at ``pose`` and return the map (3, H, W) in the robot's heading frame."""
        m = self.cfg["map"]
        base = self.scan_to_base(np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3))
        base = base[np.isfinite(base).all(axis=1)]
        rng = np.linalg.norm(base - self.mount_pos, axis=1)
        base = base[rng < m["max_range"]]
        limit = self.cfg["lidar"].get("max_points")
        if limit and len(base) > limit:  # thin to the simulated lidar's density (the map's max per cell depends on it)
            base = base[self.rng.choice(len(base), limit, replace=False)]
        self.last_scan_base = base
        world = base @ pose.rotation().T + np.array([pose.x, pose.y, pose.z])
        dev = self.device
        points = torch.as_tensor(world, dtype=torch.float32, device=dev)[None]
        local = torch.as_tensor(base, dtype=torch.float32, device=dev)[None]
        ground = torch.tensor([pose.z - m["base_height"]], dtype=torch.float32, device=dev)
        points = drop_returns(points, local, ground, tuple(map(tuple, m["crop_box"])), m["drop_above"])
        root = torch.tensor([[pose.x, pose.y, pose.z]], dtype=torch.float32, device=dev)
        yaw = torch.tensor([pose.yaw], dtype=torch.float32, device=dev)
        out = self.map.update(points, root, yaw, ground, dt, m["memory_s"], tuple(m["z_range"]), m["unknown_value"])
        return out[0].cpu().numpy()

    def proprio(self, pose: Pose, velocity_b: np.ndarray) -> np.ndarray:
        """goal (heading frame), velocity (body frame: forward, lateral, yaw rate), projected gravity, last action."""
        if self.goal is None:
            raise RuntimeError("no goal")
        dx, dy = self.goal[0] - pose.x, self.goal[1] - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        goal_b = np.array([c * dx + s * dy, -s * dx + c * dy])
        gravity_b = pose.rotation().T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([goal_b, np.asarray(velocity_b, dtype=np.float64), gravity_b, self.last_action])

    # -- commands -----------------------------------------------------------------------------------------------------

    def to_command(self, action: np.ndarray) -> np.ndarray:
        """Policy output -> velocity command, as in training (tanh into the command ranges)."""
        return self.mid + self.half * np.tanh(action)

    def guarded(self, command: np.ndarray) -> tuple[np.ndarray, bool]:
        """Stop translation (keep turning) if lidar returns lie in the box the footprint sweeps in the next 0.6 s
        (at least 0.25 m), between 5 cm and 1 m above the ground -- a last line of defence, independent of the
        policy."""
        if not self.guard or self.last_scan_base is None or len(self.last_scan_base) == 0:
            return command, False
        g = self.cfg["guard"]
        hx, hy = g["half_length"], g["half_width"]
        pts = self.last_scan_base
        z = pts[:, 2] + self.cfg["map"]["base_height"]
        pts = pts[(z > 0.05) & (z < 1.0)]
        vx, vy = command[0], command[1]
        reach_x = max(abs(vx) * g["horizon"], g["min_reach"]) if abs(vx) > 0.05 else 0.0
        reach_y = max(abs(vy) * g["horizon"], g["min_reach"]) if abs(vy) > 0.05 else 0.0
        x0 = -hx - (reach_x if vx < 0 else 0.0)
        x1 = hx + (reach_x if vx > 0 else 0.0)
        y0 = -hy - (reach_y if vy < 0 else 0.0)
        y1 = hy + (reach_y if vy > 0 else 0.0)
        inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
        # returns inside the footprint itself are the robot (the crop box), not obstacles
        outside_body = ~((np.abs(pts[:, 0]) < hx) & (np.abs(pts[:, 1]) < hy))
        if int((inside & outside_body).sum()) >= g["min_points"]:
            return np.array([0.0, 0.0, command[2]]), True
        return command, False

    def step(self, points_lidar: np.ndarray, pose: Pose, velocity_b: np.ndarray, stamp: float) -> tuple[np.ndarray, dict]:
        """One policy step for a new scan: the command (forward m/s, lateral m/s, yaw rate rad/s) and diagnostics."""
        dt = 0.1 if self.last_stamp is None else float(np.clip(stamp - self.last_stamp, 0.02, 0.5))
        self.last_stamp = stamp
        height_map = self.update_map(points_lidar, pose, dt)
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
        command, stopped = self.guarded(command)
        return command, {**info, "state": "guard stop" if stopped else "driving", "obs": obs, "action": action}

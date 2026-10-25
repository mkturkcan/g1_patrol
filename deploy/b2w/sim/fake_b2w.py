"""A simulated Unitree B2-W on ROS 2, to try the navigation stack without the robot (software in the loop).

It talks like the robot: it takes sport-mode requests on ``/api/sport/request`` (Move, StopMove; or
``geometry_msgs/Twist`` on ``/cmd_vel``), publishes its state on ``/sportmodestate``, the remote on
``/wirelesscontroller`` (the dead-man button held, unless ``deadman_held:=false``) and the RS-Helios-5515's point cloud
on ``/rslidar_points`` (frame ``rslidar``, ray-cast in a JSON world of boxes and upright cylinders from
``export_world.py``, with optional walking people). It moves with the controller model the policy was trained with
(:mod:`g1_patrol.controller_model`: latency, response time, acceleration bounds) and reports when its footprint touches
a structure or a person (``/fake_b2w/contacts``) and its true pose (``/fake_b2w/pose``).

Usage (with unitree_ros2's messages built and sourced):
    python3 -m b2w.sim.fake_b2w --ros-args -p world:=deploy/b2w/sim/lab_world.json -p start:="[-6.5, 0.0, 0.0]"
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Int32
from unitree_api.msg import Request
from unitree_go.msg import SportModeState, WirelessController

from g1_patrol.b2w import FOOTPRINT, LIDAR_MOUNT, LIDARS
from g1_patrol.controller_model import track
from ..params import declare


def helios_rays(model: str = "helios_5515", horizontal_res: float = 1.0) -> np.ndarray:
    """Unit ray directions (k, 3) of a RoboSense lidar in its frame."""
    channels, (lo, hi), _ = LIDARS[model]
    el = np.radians(np.linspace(lo, hi, channels))
    az = np.radians(np.arange(-180.0, 180.0, horizontal_res))
    e, a = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1).reshape(-1, 3)


def raycast(origin: np.ndarray, dirs: np.ndarray, boxes: np.ndarray, cylinders: np.ndarray, max_range: float) -> np.ndarray:
    """Distance along each ray (k,) to the floor (z = 0), oriented boxes (cx, cy, yaw, hx, hy, z0, z1) and upright
    cylinders (x, y, r, z0, z1); inf where nothing is hit within ``max_range``."""
    k = len(dirs)
    t = np.full(k, np.inf)
    down = dirs[:, 2] < -1e-6
    t[down] = -origin[2] / dirs[down, 2]
    if len(boxes):
        cx, cy, yaw, hx, hy, z0, z1 = (boxes[:, i][None] for i in range(7))
        c, s = np.cos(yaw), np.sin(yaw)
        ox, oy = origin[0] - cx, origin[1] - cy
        lox, loy = c * ox + s * oy, -s * ox + c * oy  # origin in the box frame
        ldx = c * dirs[:, :1] + s * dirs[:, 1:2]
        ldy = -s * dirs[:, :1] + c * dirs[:, 1:2]
        ldz = np.broadcast_to(dirs[:, 2:3], ldx.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1x, t2x = (-hx - lox) / ldx, (hx - lox) / ldx
            t1y, t2y = (-hy - loy) / ldy, (hy - loy) / ldy
            t1z, t2z = (z0 - origin[2]) / ldz, (z1 - origin[2]) / ldz
        tmin = np.maximum.reduce([np.minimum(t1x, t2x), np.minimum(t1y, t2y), np.minimum(t1z, t2z)])
        tmax = np.minimum.reduce([np.maximum(t1x, t2x), np.maximum(t1y, t2y), np.maximum(t1z, t2z)])
        hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
        tb = np.where(hit, np.maximum(tmin, 0.0), np.inf)
        t = np.minimum(t, tb.min(axis=1))
    if len(cylinders):
        x, y, r, z0, z1 = (cylinders[:, i][None] for i in range(5))
        ox, oy = origin[0] - x, origin[1] - y
        dx, dy, dz = dirs[:, :1], dirs[:, 1:2], dirs[:, 2:3]
        a = dx * dx + dy * dy
        b = 2 * (ox * dx + oy * dy)
        cc = ox * ox + oy * oy - r * r
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            tc = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
        zc = origin[2] + tc * dz
        side = (disc >= 0) & (tc > 0) & (zc >= z0) & (zc <= z1)
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = (z1 - origin[2]) / dz  # the top
        px, py = ox + tt * dx, oy + tt * dy
        top = (tt > 0) & (px * px + py * py <= r * r)
        t = np.minimum(t, np.where(side, tc, np.inf).min(axis=1))
        t = np.minimum(t, np.where(top, tt, np.inf).min(axis=1))
    return np.where(t < max_range, t, np.inf)


def footprint_gap(x: float, y: float, yaw: float, boxes: np.ndarray, cylinders: np.ndarray, overhead: float = 1.0) -> float:
    """Smallest gap (m) between the rectangular footprint and the structures lower than ``overhead``."""
    hx, hy = FOOTPRINT
    u, v = np.meshgrid(np.linspace(-hx, hx, 23), np.linspace(-hy, hy, 13), indexing="ij")
    edge = (np.abs(u) >= hx - 1e-9) | (np.abs(v) >= hy - 1e-9)
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = x + c * u[edge] - s * v[edge], y + s * u[edge] + c * v[edge]
    gap = 1e3
    inside = False
    for bx, by, byaw, bhx, bhy, z0, _ in boxes:
        if z0 >= overhead:
            continue
        bc, bs = math.cos(byaw), math.sin(byaw)
        lx, ly = bc * (px - bx) + bs * (py - by), -bs * (px - bx) + bc * (py - by)
        gap = min(gap, float(np.min(np.hypot(np.maximum(np.abs(lx) - bhx, 0), np.maximum(np.abs(ly) - bhy, 0)))))
        lcx, lcy = bc * (x - bx) + bs * (y - by), -bs * (x - bx) + bc * (y - by)
        inside |= abs(lcx) < bhx and abs(lcy) < bhy
    for cx, cy, r, z0, _ in cylinders:
        if z0 < overhead:
            gap = min(gap, float(np.min(np.hypot(px - cx, py - cy))) - r)
    return -1.0 if inside else gap


class People:
    """People walking back and forth along paths (lists of points) at their speeds."""

    def __init__(self, paths: list, speeds: list):
        self.paths = [np.asarray(p, dtype=float) for p in paths]
        self.speeds = speeds
        self.s = [0.0] * len(paths)
        self.dir = [1.0] * len(paths)

    def step(self, dt: float):
        for i, p in enumerate(self.paths):
            length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            self.s[i] += self.dir[i] * self.speeds[i] * dt
            if self.s[i] > length or self.s[i] < 0:
                self.dir[i] *= -1
                self.s[i] = min(max(self.s[i], 0.0), length)

    def positions(self) -> np.ndarray:
        out = []
        for i, p in enumerate(self.paths):
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            cum = np.r_[0, np.cumsum(seg)]
            k = min(int(np.searchsorted(cum, self.s[i], side="right")) - 1, len(seg) - 1)
            f = (self.s[i] - cum[k]) / max(seg[k], 1e-9)
            out.append(p[k] + f * (p[k + 1] - p[k]))
        return np.array(out).reshape(-1, 2)


class FakeB2W(Node):
    def __init__(self):
        super().__init__("fake_b2w")
        p = lambda name, default: declare(self, name, default)  # noqa: E731
        world = json.loads(open(p("world", "deploy/b2w/sim/lab_world.json")).read())
        self.boxes = np.array(world["boxes"], dtype=float).reshape(-1, 7)
        self.cylinders = np.array(world["circles"], dtype=float).reshape(-1, 5)
        start = [float(v) for v in p("start", [-6.5, 0.0, 0.0])]
        self.x, self.y, self.yaw = start
        self.base_height = float(p("base_height", 0.65))
        self.deadman_held = bool(p("deadman_held", True))
        people = p("people", "")  # JSON: [{"path": [[x, y], ...], "speed": 1.0}, ...]
        spec = json.loads(people) if people else []
        self.people = People([q["path"] for q in spec], [q["speed"] for q in spec])
        self.dt = 0.02
        self.delay = int(round(float(p("latency", 0.08)) / self.dt))
        self.queue = [np.zeros(3)] * (self.delay + 1)
        self.vel = torch.zeros(1, 3)
        self.gain, self.deadband = torch.ones(1, 3), torch.zeros(1, 3)
        self.tau = torch.tensor([[0.25, 0.25, 0.15]])
        self.accel = torch.tensor([[2.0, 1.5, 4.0]])
        self.command = np.zeros(3)
        self.command_time = 0.0
        self.rays = helios_rays()
        self.contacts = 0
        self.robot_moving_contacts = 0
        self.touching = False
        self.create_subscription(Request, "/api/sport/request", self.on_request, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_twist, 10)
        self.state_pub = self.create_publisher(SportModeState, "/sportmodestate", qos_profile_sensor_data)
        self.remote_pub = self.create_publisher(WirelessController, "/wirelesscontroller", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/rslidar_points", qos_profile_sensor_data)
        self.contact_pub = self.create_publisher(Int32, "/fake_b2w/contacts", 10)
        self.moving_contact_pub = self.create_publisher(Int32, "/fake_b2w/robot_moving_contacts", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/fake_b2w/pose", 10)
        self.create_timer(self.dt, self.physics)
        self.create_timer(0.1, self.lidar)
        self.create_timer(0.05, self.remote)
        self.get_logger().info(f"fake B2-W in {world.get('source', 'a world')}: {len(self.boxes)} boxes, "
                               f"{len(self.cylinders)} cylinders, {len(self.people.paths)} people, start {start}")

    def on_request(self, msg: Request):
        if msg.header.identity.api_id == 1008:
            p = json.loads(msg.parameter)
            self.command = np.array([p["x"], p["y"], p["z"]], dtype=float)
            self.command_time = time.monotonic()
        elif msg.header.identity.api_id == 1003:
            self.command = np.zeros(3)

    def on_twist(self, msg: Twist):
        self.command = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.command_time = time.monotonic()

    def physics(self):
        if time.monotonic() - self.command_time > 0.5:  # commands stopped coming: stop, as the sport mode does
            self.command = np.zeros(3)
        self.queue = [self.command.copy()] + self.queue[:-1]
        cmd = torch.tensor(self.queue[-1], dtype=torch.float32)[None]
        self.vel = track(self.vel, cmd, self.dt, self.gain, self.deadband, self.tau, self.accel, torch.zeros(1))
        vx, vy, wz = (float(v) for v in self.vel[0])
        self.yaw += wz * self.dt
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * self.dt
        self.y += (s * vx + c * vy) * self.dt
        self.people.step(self.dt)
        gap = footprint_gap(self.x, self.y, self.yaw, self.boxes, self.cylinders)
        people = self.people.positions()
        if len(people):
            hx, hy = FOOTPRINT
            rel = people - np.array([self.x, self.y])
            lx, ly = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
            gap = min(gap, float(np.min(np.hypot(np.maximum(np.abs(lx) - hx, 0), np.maximum(np.abs(ly) - hy, 0)))) - 0.25)
        touching = gap < 0.0
        if touching and not self.touching:
            fastest = math.hypot(vx, vy) + abs(wz) * math.hypot(*FOOTPRINT)
            moving = fastest > 0.1
            self.contacts += 1
            self.robot_moving_contacts += moving
            self.get_logger().warning(f"CONTACT #{self.contacts} at ({self.x:.2f}, {self.y:.2f}): robot "
                                      f"{'moving' if moving else 'stopped'} (fastest point {fastest:.2f} m/s)")
        self.touching = touching
        msg = SportModeState()
        msg.position = [float(self.x), float(self.y), float(self.base_height)]
        msg.imu_state.quaternion = [float(math.cos(self.yaw / 2)), 0.0, 0.0, float(math.sin(self.yaw / 2))]
        msg.imu_state.gyroscope = [0.0, 0.0, float(wz)]
        msg.velocity = [float(c * vx - s * vy), float(s * vx + c * vy), 0.0]
        msg.yaw_speed = float(wz)
        msg.body_height = float(self.base_height)
        self.state_pub.publish(msg)
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y = float(self.x), float(self.y)
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.pose_pub.publish(pose)
        self.contact_pub.publish(Int32(data=self.contacts))
        self.moving_contact_pub.publish(Int32(data=self.robot_moving_contacts))

    def lidar(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        mx, my, mz = LIDAR_MOUNT
        origin = np.array([self.x + c * mx - s * my, self.y + s * mx + c * my, self.base_height + mz])
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        dirs = self.rays @ rot.T
        people = self.people.positions()
        cyl = np.concatenate([self.cylinders, np.c_[people, np.full(len(people), 0.25), np.zeros(len(people)),
                                                    np.full(len(people), 1.7)]]) if len(people) else self.cylinders
        t = raycast(origin, dirs, self.boxes, cyl, LIDARS["helios_5515"][2])
        ok = np.isfinite(t)
        pts = (self.rays[ok] * t[ok, None]).astype(np.float32)
        pts += np.random.default_rng().normal(0.0, 0.01, pts.shape).astype(np.float32)  # range noise
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "rslidar"
        msg.height, msg.width = 1, len(pts)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 12, 12 * len(pts), True
        msg.data = pts.tobytes()
        self.cloud_pub.publish(msg)

    def remote(self):
        msg = WirelessController()
        msg.keys = 1 << 1 if self.deadman_held else 0  # L1
        self.remote_pub.publish(msg)


def main():
    rclpy.init()
    node = FakeB2W()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

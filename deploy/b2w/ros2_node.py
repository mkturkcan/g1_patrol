"""ROS 2 node that drives the Unitree B2-W with the navigation policy (runs on the robot's Jetson Orin NX).

Inputs:

* the lidar (``sensor_msgs/PointCloud2`` on ``lidar_topic``, default ``/rslidar_points`` from RoboSense's
  ``rslidar_sdk``; the mount comes from ``deploy.json``, not TF);
* the depth images of the RealSense D435i cameras named in ``deploy.json`` (``<name>_depth_topic``,
  ``<name>_info_topic``; 16-bit millimeters from ``realsense2_camera``) -- optional: the policy was trained with
  cameras that fail, and runs without them;
* odometry (``nav_msgs/Odometry`` on ``odom_topic``: base_link in the odometry frame, twist in base_link; from
  ``b2w.sportstate_to_odom``);
* the route (``nav_msgs/Path`` on ``path_topic``, default ``/plan``, from ``b2w.planner_node`` on the SLAM map, or
  any global planner), brought into the odometry frame with TF (``map`` -> ``odom`` from SLAM) when it arrives; an
  empty path means "stand". Without a planner, ``goal_topic`` (if set) takes goals reached in a straight line;
* the remote (``unitree_go/msg/WirelessController`` on ``remote_topic``): the policy drives only while the
  ``deadman`` button is held, and stops sending as soon as a stick is moved (the operator takes over).

One policy step per lidar scan (10 Hz): the height map (with the floor measured in the scan), the policy, the speed
cap ``max_speed`` (which the policy is told), the safety shield (:mod:`g1_patrol.shield`). Commands go to the
sport-mode controller as ``unitree_api/msg/Request`` on ``/api/sport/request`` (``output:=ros``, the default; needs
unitree_ros2's messages), through ``unitree_sdk2py``'s sport client (``output:=sdk``) or as ``geometry_msgs/Twist`` on
``cmd_vel`` (``output:=twist``, for a bridge or a simulator).

Modes (parameter ``mode``):

* ``shadow`` (default): compute and publish what the policy would command (``~/command``, ``~/state``, markers) but
  send nothing -- drive the robot by hand and watch;
* ``drive``: send the commands; the robot is stopped when the lidar or odometry are older than ``timeout`` s, when the
  dead-man button is released, when a stick moves, while ``~/stop`` (``std_msgs/Bool``) is true, and when the goal is
  reached;
* ``sysid``: play the command profile ``profile`` (CSV from ``scripts/b2w/sysid.py profile``) and log commands and
  odometry for ``scripts/b2w/sysid.py fit`` (the dead-man button must be held).

``record`` (a path) stores every scan with its pose, velocity, route and command (``.npz``) for
``scripts/b2w/replay.py``. Started with the rest of the stack by ``launch/b2w_navigation.launch.py``.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from g1_patrol.route import path_at
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, Float32MultiArray, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .navigator import Navigator, OnnxPolicy, Pose, depth_image_points
from .params import declare

BUTTONS = {name: 1 << k for k, name in enumerate(
    ["R1", "L1", "start", "select", "R2", "L2", "F1", "F3", "A", "B", "X", "Y", "up", "right", "down", "left"])}
"""Bits of ``WirelessController.keys`` (unitree_sdk2's remote layout)."""


def _quat_to_rpy(q) -> tuple[float, float, float]:
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    """The x, y, z fields of a point cloud (k, 3), without sensor_msgs_py (not in ROS 2 Foxy)."""
    types = {7: "f4", 8: "f8"}
    fields = {f.name: f for f in msg.fields}
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": [types[fields[c].datatype] for c in "xyz"],
                      "offsets": [fields[c].offset for c in "xyz"], "itemsize": msg.point_step})
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


class RosSportOutput:
    """Velocity commands as unitree_api requests on /api/sport/request (the sport service, Move = 1008)."""

    MOVE, STOP_MOVE = 1008, 1003

    def __init__(self, node: Node, topic: str):
        from unitree_api.msg import Request

        self.Request = Request
        self.pub = node.create_publisher(Request, topic, 10)

    def _request(self, api_id: int, parameter: str = ""):
        req = self.Request()
        req.header.identity.id = time.monotonic_ns()
        req.header.identity.api_id = api_id
        req.parameter = parameter
        self.pub.publish(req)

    def send(self, vx: float, vy: float, wz: float):
        self._request(self.MOVE, json.dumps({"x": float(vx), "y": float(vy), "z": float(wz)}))

    def stop(self):
        self._request(self.STOP_MOVE)


class SdkSportOutput:
    """Velocity commands through unitree_sdk2py's sport client (DDS, no ROS messages needed)."""

    def __init__(self, network_interface: str):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        try:
            from unitree_sdk2py.b2.sport.sport_client import SportClient  # B2 and B2-W
        except ImportError:  # older SDKs only have the Go2 client, whose Move() has the same signature
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        ChannelFactoryInitialize(0, network_interface)
        self.client = SportClient()
        self.client.SetTimeout(0.2)
        self.client.Init()

    def send(self, vx: float, vy: float, wz: float):
        self.client.Move(float(vx), float(vy), float(wz))

    def stop(self):
        self.client.StopMove()


class TwistOutput:
    """Velocity commands as geometry_msgs/Twist (a bridge, or a simulator)."""

    def __init__(self, node: Node, topic: str):
        self.pub = node.create_publisher(Twist, topic, 10)

    def send(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.pub.publish(msg)

    def stop(self):
        self.send(0.0, 0.0, 0.0)


class B2WNavNode(Node):
    def __init__(self):
        super().__init__("b2w_nav")
        p = lambda name, default: declare(self, name, default)  # noqa: E731
        config = p("config", "deploy.json")
        policy = p("policy", "policy.onnx")
        self.mode = p("mode", "shadow")
        self.timeout = float(p("timeout", 0.3))
        max_speed = float(p("max_speed", 0.5))
        output = p("output", "ros")
        record = p("record", "")
        self.nav = Navigator(config, OnnxPolicy(policy), device=p("device", "cpu"), max_speed=max_speed,
                             guard=bool(p("guard", True)))
        lo, hi = self.nav.cfg["speed_limit_range"]
        if not lo <= max_speed <= hi:
            self.get_logger().warning(f"max_speed {max_speed} m/s is outside the caps the policy was trained with "
                                      f"({lo}-{hi} m/s); it is told {self.nav.speed_limit} m/s")
        self.deadman = p("deadman", "L1")
        self.takeover = float(p("takeover_threshold", 0.15))
        self.odom: Odometry | None = None
        self.odom_time = self.scan_time = 0.0
        self.remote_time = 0.0
        self.deadman_held = self.deadman == "none"
        self.operator = False
        self.sending = False
        self.state = "starting"
        self.records: list[dict] = []
        self.record_path = Path(record) if record else None
        self.create_subscription(PointCloud2, p("lidar_topic", "/rslidar_points"), self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, p("odom_topic", "/odom"), self.on_odom, 20)
        self.create_subscription(NavPath, p("path_topic", "/plan"), self.on_path, 10)
        goal_topic = p("goal_topic", "")
        if goal_topic:
            self.create_subscription(PoseStamped, goal_topic, self.on_goal, 1)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        # depth cameras (RealSense D435i: realsense2_camera's depth image and its camera info)
        self.depth: dict[str, tuple[float, np.ndarray]] = {}
        self.intrinsics: dict[str, tuple[float, float, float, float]] = {}
        for name in self.nav.cameras:
            self.create_subscription(Image, p(f"{name}_depth_topic", f"/{name}_camera/depth/image_rect_raw"),
                                     lambda msg, n=name: self.on_depth(n, msg), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, p(f"{name}_info_topic", f"/{name}_camera/depth/camera_info"),
                                     lambda msg, n=name: self.on_info(n, msg), 1)
        # the remote: dead-man button and operator takeover
        remote_topic = p("remote_topic", "/wirelesscontroller")
        if self.deadman != "none" or self.takeover > 0:
            try:
                from unitree_go.msg import WirelessController

                self.create_subscription(WirelessController, remote_topic, self.on_remote, 20)
            except ImportError:
                self.get_logger().error("unitree_go messages not found (source unitree_ros2): no dead-man button, the"
                                        " robot will not drive" if self.deadman != "none" else
                                        "unitree_go messages not found: no operator takeover from the remote")
        if self.deadman not in BUTTONS and self.deadman != "none":
            raise ValueError(f"deadman must be one of {sorted(BUTTONS)} or 'none'")
        # emergency stop: True stops the robot until False is published (latched)
        self.stopped = bool(p("start_stopped", False))
        self.create_subscription(Bool, "~/stop", self.on_stop, 10)
        self.command_pub = self.create_publisher(Float32MultiArray, "~/command", 10)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.out = None
        if self.mode in ("drive", "sysid"):
            if output == "sdk":
                self.out = SdkSportOutput(p("network_interface", "eth0"))
            elif output == "twist":
                self.out = TwistOutput(self, p("cmd_vel_topic", "/cmd_vel"))
            else:
                self.out = RosSportOutput(self, p("sport_request_topic", "/api/sport/request"))
        self.create_timer(0.05, self.watchdog)
        self.create_timer(5.0, self.health)
        if self.mode == "sysid":
            self.start_sysid(p("profile", "profile.csv"), p("sysid_log", "sysid_log.csv"))
        self.get_logger().info(
            f"B2-W navigation: mode {self.mode}, output {output if self.out else 'none'}, max speed {max_speed} m/s, "
            f"dead-man button {self.deadman}, cameras {list(self.nav.cameras) or 'none'}")

    # -- inputs -------------------------------------------------------------------------------------------------------

    def on_odom(self, msg: Odometry):
        self.odom = msg
        self.odom_time = time.monotonic()

    def on_info(self, name: str, msg: CameraInfo):
        self.intrinsics[name] = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_depth(self, name: str, msg: Image):
        if name not in self.intrinsics or msg.encoding not in ("16UC1", "mono16"):
            return
        depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        c = self.nav.cameras[name][2]
        self.depth[name] = (time.monotonic(), depth_image_points(depth, *self.intrinsics[name], c["width"], c["height"]))

    def on_remote(self, msg):
        self.remote_time = time.monotonic()
        if self.deadman != "none":
            self.deadman_held = bool(msg.keys & BUTTONS[self.deadman])
        sticks = max(abs(msg.lx), abs(msg.ly), abs(msg.rx), abs(msg.ry))
        operator = self.takeover > 0 and sticks > self.takeover
        if operator and not self.operator:
            self.get_logger().warning("operator takeover (stick moved): the policy stops sending")
        self.operator = operator
        if (not self.deadman_held or self.operator) and self.sending:
            self.halt()

    def on_stop(self, msg: Bool):
        self.stopped = bool(msg.data)
        self.get_logger().warning("STOP" if self.stopped else "stop released")
        if self.stopped:
            self.halt()

    def to_odom(self, frame: str, points: np.ndarray) -> np.ndarray | None:
        """Points (k, 2) in ``frame`` -> the odometry frame (TF, frames level); None if the transform is missing."""
        odom = self.odom.header.frame_id if self.odom is not None else ""
        if not frame or not odom or frame == odom:
            return points
        try:
            t = self.tf.lookup_transform(odom, frame, Time())
        except Exception as e:  # noqa: BLE001 (tf2's lookup, connectivity and extrapolation errors)
            self.get_logger().warning(f"no transform {frame} -> {odom}: {e}", throttle_duration_sec=2.0)
            return None
        yaw = _quat_to_rpy(t.transform.rotation)[2]
        c, s = math.cos(yaw), math.sin(yaw)
        return points @ np.array([[c, s], [-s, c]]) + np.array([t.transform.translation.x, t.transform.translation.y])

    def on_path(self, msg: NavPath):
        if not msg.poses:  # no route: stand
            if self.nav.goal is not None:
                self.get_logger().info("empty route: standing")
            self.nav.clear()
            return
        points = self.to_odom(msg.header.frame_id, np.array([[q.pose.position.x, q.pose.position.y] for q in msg.poses]))
        if points is not None:
            self.nav.set_path(points)

    def on_goal(self, msg: PoseStamped):
        goal = self.to_odom(msg.header.frame_id, np.array([[msg.pose.position.x, msg.pose.position.y]]))
        if goal is not None:
            self.nav.set_goal(*goal[0])
            self.get_logger().info(f"goal ({goal[0, 0]:.2f}, {goal[0, 1]:.2f}) in the odometry frame, straight line")

    def pose(self) -> tuple[Pose, np.ndarray]:
        o = self.odom
        roll, pitch, yaw = _quat_to_rpy(o.pose.pose.orientation)
        pose = Pose(o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z, roll, pitch, yaw)
        t = o.twist.twist
        return pose, np.array([t.linear.x, t.linear.y, t.angular.z])

    # -- the policy step ----------------------------------------------------------------------------------------------

    def on_scan(self, msg: PointCloud2):
        self.scan_time = time.monotonic()
        if self.odom is None or self.mode == "sysid":
            return
        points = cloud_xyz(msg)
        pose, velocity = self.pose()
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        now = time.monotonic()
        depth = {name: pts for name, (t, pts) in self.depth.items() if now - t < self.timeout}
        command, info = self.nav.step(points, pose, velocity, stamp, depth)
        self.state = info.get("state", "")
        self.command_pub.publish(Float32MultiArray(data=[float(c) for c in command]))
        if self.mode == "drive" and self.out is not None:
            if self.state == "reached" and self.sending:
                self.get_logger().info("goal reached")
                self.halt()
            elif self.may_drive() and self.state not in ("no goal", "reached"):
                self.out.send(*command)
                self.sending = True
            elif self.sending:
                self.halt()
        shield = info.get("shield", {})
        self.state_pub.publish(String(data=json.dumps({
            "state": self.state, "sending": self.sending, "deadman": self.deadman_held, "operator": self.operator,
            "stopped": self.stopped, "command": [round(float(c), 3) for c in command],
            "policy_command": [round(float(c), 3) for c in info.get("policy_command", command)],
            "yaw": round(pose.yaw, 3),
            "goal_distance": round(info.get("goal_distance", -1.0), 2), "floor": round(self.nav.floor_z, 3),
            "cameras": sorted(depth), "points": int(len(points)), "shield_scale": round(shield.get("scale", 1.0), 2),
            "min_gap": round(min(shield.get("min_gap", 1e3), 1e3), 2),
            "moving_gap": round(min(shield.get("min_moving_gap", 1e3), 1e3), 2),
            "crawl": bool(shield.get("crawling", 0.0)), "alternative": int(shield.get("alternative", -1))})))
        self.publish_markers(pose, msg.header.stamp)
        if self.record_path is not None:
            self.records.append({"stamp": stamp, "points": points.astype(np.float32),
                "depth": {k: v.astype(np.float32) for k, v in depth.items()}, "pose": np.array(
                [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]), "velocity": velocity, "command": command,
                "goal": self.nav.goal if self.nav.goal is not None else np.full(2, np.nan),
                "route": self.nav.route[0][0].numpy() if self.nav.route is not None else np.zeros((0, 2)),
                "route_version": self.nav.route_version, "max_speed": self.nav.max_speed})

    def may_drive(self) -> bool:
        return (not self.stopped and self.fresh() and self.deadman_held and not self.operator
                and (self.deadman == "none" or time.monotonic() - self.remote_time < 0.5))

    def halt(self):
        if self.out is not None:
            self.out.stop()
        self.sending = False

    def fresh(self) -> bool:
        now = time.monotonic()
        return now - self.scan_time < self.timeout and now - self.odom_time < self.timeout

    def watchdog(self):
        if self.sending and not self.may_drive():
            self.get_logger().warning("stopping: " + ("stop requested" if self.stopped else "stale lidar or odometry"
                                      if not self.fresh() else "dead-man button released" if not self.deadman_held
                                      else "operator took over" if self.operator else "remote silent"))
            self.halt()

    def health(self):
        now = time.monotonic()
        problems = []
        if now - self.scan_time > 1.0:
            problems.append("no lidar scans")
        if now - self.odom_time > 1.0:
            problems.append("no odometry")
        if self.deadman != "none" and now - self.remote_time > 1.0:
            problems.append("no remote messages (dead-man button)")
        if problems:
            self.get_logger().warning("waiting: " + ", ".join(problems))

    def publish_markers(self, pose: Pose, stamp):
        """Footprint and the policy's two route waypoints, for RViz (odometry frame)."""
        frame = self.odom.header.frame_id if self.odom is not None else "odom"
        markers = MarkerArray()
        fp = Marker()
        fp.header.frame_id, fp.header.stamp, fp.ns, fp.id = frame, stamp, "footprint", 0
        fp.type, fp.scale.x = Marker.LINE_STRIP, 0.03
        fp.color.r, fp.color.g, fp.color.b, fp.color.a = 0.1, 0.6, 1.0, 1.0
        hx, hy = self.nav.shield_cfg.half_length, self.nav.shield_cfg.half_width
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for u, v in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy), (hx, hy)):
            fp.points.append(Point(x=pose.x + c * u - s * v, y=pose.y + s * u + c * v, z=pose.z + self.nav.floor_z))
        markers.markers.append(fp)
        if self.nav.route is not None and self.nav.route_s is not None:
            wp = Marker()
            wp.header.frame_id, wp.header.stamp, wp.ns, wp.id = frame, stamp, "waypoints", 1
            wp.type, wp.scale.x, wp.scale.y, wp.scale.z = Marker.SPHERE_LIST, 0.15, 0.15, 0.15
            wp.color.r, wp.color.g, wp.color.b, wp.color.a = 1.0, 0.3, 0.8, 1.0
            pts, cum = self.nav.route
            s_ahead = self.nav.route_s + self.nav.lookahead
            ahead, _ = path_at(pts.expand(len(s_ahead), -1, -1), cum.expand(len(s_ahead), -1), s_ahead)
            for x, y in ahead.numpy():
                wp.points.append(Point(x=float(x), y=float(y), z=pose.z + self.nav.floor_z + 0.1))
            markers.markers.append(wp)
        self.marker_pub.publish(markers)

    # -- system identification ----------------------------------------------------------------------------------------

    def start_sysid(self, profile: str, log: str):
        rows = list(csv.DictReader(open(profile)))
        self.profile = np.array([[float(r["t"]), float(r["vx"]), float(r["vy"]), float(r["wz"])] for r in rows])
        self.sysid_log = open(log, "w", newline="")
        self.sysid_writer = csv.writer(self.sysid_log)
        self.sysid_writer.writerow(["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "x", "y", "yaw"])
        self.sysid_start: float | None = None
        self.create_timer(0.02, self.sysid_tick)
        self.get_logger().info(f"sysid: {len(rows)} commands over {self.profile[-1, 0]:.0f} s -> {log}; "
                               f"hold the dead-man button ({self.deadman}) to play")

    def sysid_tick(self):
        if not (self.deadman_held and not self.operator and self.odom is not None and time.monotonic() - self.odom_time < self.timeout):
            if self.out is not None and self.sending:
                self.halt()
            return
        if self.sysid_start is None:
            self.sysid_start = time.monotonic()
        t = time.monotonic() - self.sysid_start
        if t > self.profile[-1, 0]:
            self.halt()
            self.sysid_log.close()
            self.get_logger().info("sysid done")
            rclpy.shutdown()
            return
        k = int(np.searchsorted(self.profile[:, 0], t, side="right")) - 1
        cmd = self.profile[max(k, 0), 1:]
        self.out.send(*cmd)
        self.sending = True
        pose, velocity = self.pose()
        self.sysid_writer.writerow([f"{t:.4f}", *(f"{c:.4f}" for c in cmd), *(f"{v:.4f}" for v in velocity),
                                    f"{pose.x:.4f}", f"{pose.y:.4f}", f"{pose.yaw:.4f}"])

    def save(self):
        if self.record_path is not None and self.records:
            keys = self.records[0].keys()
            np.savez_compressed(self.record_path, **{k: np.array([r[k] for r in self.records], dtype=object if k in ("points", "depth", "route") else None) for k in keys})
            self.get_logger().info(f"recorded {len(self.records)} scans -> {self.record_path}")


def main():
    rclpy.init()
    node = B2WNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.out is not None:
            node.out.stop()
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

"""ROS 2 node that drives the Unitree B2-W with the navigation policy (runs on the robot's Jetson Orin NX).

Subscribes to the lidar (``sensor_msgs/PointCloud2``, e.g. ``/rslidar_points`` from RoboSense's ``rslidar_sdk``), the
depth images of the RealSense D435i cameras named in ``deploy.json`` (``<name>_depth_topic``, ``<name>_info_topic``;
16-bit, millimeters, from ``realsense2_camera``),
odometry (``nav_msgs/Odometry``: base_link in the odometry frame, twist in base_link) and the route to follow:

* a global planner's path (``nav_msgs/Path`` on ``path_topic``, default ``/plan``), e.g. from Nav2's planner server;
* with ``planner_action`` (e.g. ``/compute_path_to_pose``, Nav2's planner server with a map and localization), the
  node asks the planner for a path to each goal (``geometry_msgs/PoseStamped`` on ``goal_topic``, e.g. RViz's "2D Goal
  Pose") and again every ``replan_period`` s while driving; the robot stands until the first path arrives;
* without a planner, a goal is reached in a straight line (open spaces only).

Paths and goals in another frame than the odometry's (e.g. ``map``) are brought into it with TF (``map`` -> ``odom``
from the localization) when they arrive. One policy step per lidar scan; the command goes to the robot's sport-mode
controller, either through ``unitree_sdk2py``'s sport client (``--ros-args -p output:=sdk``) or as
``geometry_msgs/Twist`` on ``cmd_vel`` for a bridge (``output:=twist``).

Modes (parameter ``mode``):

* ``shadow`` (default): compute and publish the commands the policy would give on ``~/command`` (and log them), but
  do not move the robot -- drive it by hand and watch what the policy would do;
* ``drive``: send the commands, capped at ``max_speed`` (default 0.5 m/s; the policy was trained with caps of 0.4 to
  1.5 m/s and is told the cap) and filtered by the safety shield
  (:mod:`g1_patrol.shield`); the robot stops if lidar or odometry are older than ``timeout`` s, and while
  ``~/stop`` (``std_msgs/Bool``) is true;
* ``sysid``: play the command profile ``profile`` (CSV from ``scripts/b2w/sysid.py profile``) and log commands and
  odometry for ``scripts/b2w/sysid.py fit``.

``record`` (a path) stores every scan with its pose, velocity and the policy's command (``.npz``) for
``scripts/b2w/replay.py``.

Example::

    PYTHONPATH=<repo>/src:<repo>/deploy python3 -m b2w.ros2_node --ros-args \\
        -p config:=deploy.json -p policy:=policy.onnx -p mode:=shadow -p lidar_topic:=/rslidar_points
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32MultiArray, String
from tf2_ros import Buffer, TransformListener

from .navigator import Navigator, OnnxPolicy, Pose, depth_image_points


def _quat_to_rpy(q) -> tuple[float, float, float]:
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class SportOutput:
    """Velocity commands through unitree_sdk2py's sport client (DDS, no ROS bridge needed)."""

    def __init__(self, network_interface: str):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        try:
            from unitree_sdk2py.b2.sport.sport_client import SportClient
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
    """Velocity commands as geometry_msgs/Twist, for a bridge to the sport-mode controller."""

    def __init__(self, publisher):
        self.publisher = publisher

    def send(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.publisher.publish(msg)

    def stop(self):
        self.send(0.0, 0.0, 0.0)


class B2WNavNode(Node):
    def __init__(self):
        super().__init__("b2w_nav")
        p = self.declare_parameter
        config = p("config", "deploy.json").value
        policy = p("policy", "policy.onnx").value
        self.mode = p("mode", "shadow").value
        self.timeout = float(p("timeout", 0.3).value)
        max_speed = float(p("max_speed", 0.5).value)
        output = p("output", "twist").value
        record = p("record", "").value
        self.nav = Navigator(config, OnnxPolicy(policy), max_speed=max_speed, guard=bool(p("guard", True).value))
        self.odom: Odometry | None = None
        self.odom_time = 0.0
        self.scan_time = 0.0
        self.records: list[dict] = []
        self.record_path = Path(record) if record else None
        self.create_subscription(PointCloud2, p("lidar_topic", "/rslidar_points").value, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, p("odom_topic", "/odom").value, self.on_odom, 20)
        self.create_subscription(PoseStamped, p("goal_topic", "/goal_pose").value, self.on_goal, 1)
        self.create_subscription(NavPath, p("path_topic", "/plan").value, self.on_path, 1)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.goal_msg: PoseStamped | None = None
        self.planner, self.planning = None, False
        planner = p("planner_action", "").value
        if planner:
            from nav2_msgs.action import ComputePathToPose
            from rclpy.action import ActionClient

            self.planner = ActionClient(self, ComputePathToPose, planner)
            self.create_timer(float(p("replan_period", 1.0).value), self.replan)
        # depth cameras (RealSense D435i: realsense2_camera's aligned-to-depth image and its camera info)
        self.depth: dict[str, tuple[float, np.ndarray]] = {}
        self.intrinsics: dict[str, tuple[float, float, float, float]] = {}
        for name in self.nav.cameras:
            self.create_subscription(Image, p(f"{name}_depth_topic", f"/{name}_camera/depth/image_rect_raw").value,
                                     lambda msg, n=name: self.on_depth(n, msg), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, p(f"{name}_info_topic", f"/{name}_camera/depth/camera_info").value,
                                     lambda msg, n=name: self.on_info(n, msg), 1)
        # emergency stop: True stops the robot until False is published (latched)
        self.stopped = bool(p("start_stopped", False).value)
        self.create_subscription(Bool, "~/stop", self.on_stop, 10)
        self.command_pub = self.create_publisher(Float32MultiArray, "~/command", 10)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.out = None
        if self.mode in ("drive", "sysid"):
            if output == "sdk":
                self.out = SportOutput(p("network_interface", "eth0").value)
            else:
                self.out = TwistOutput(self.create_publisher(Twist, p("cmd_vel_topic", "cmd_vel").value, 10))
        self.create_timer(0.05, self.watchdog)
        if self.mode == "sysid":
            self.start_sysid(p("profile", "profile.csv").value, p("sysid_log", "sysid_log.csv").value)
        self.get_logger().info(f"B2-W navigation: mode {self.mode}, output {output if self.out else 'none'}, max speed {max_speed} m/s")

    # -- inputs -------------------------------------------------------------------------------------------------------

    def on_odom(self, msg: Odometry):
        self.odom = msg
        self.odom_time = time.monotonic()

    def on_info(self, name: str, msg: CameraInfo):
        self.intrinsics[name] = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_depth(self, name: str, msg: Image):
        if name not in self.intrinsics or msg.encoding not in ("16UC1", "mono16"):
            return
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        c = self.nav.cameras[name][2]
        self.depth[name] = (time.monotonic(), depth_image_points(depth, *self.intrinsics[name], c["width"], c["height"]))

    def on_stop(self, msg: Bool):
        self.stopped = bool(msg.data)
        self.get_logger().warning("STOP" if self.stopped else "stop released")
        if self.stopped and self.out is not None:
            self.out.stop()

    def to_odom(self, frame: str, points: np.ndarray) -> np.ndarray | None:
        """Points (k, 2) in ``frame`` -> the odometry frame (TF, frames level); None if the transform is missing."""
        odom = self.odom.header.frame_id if self.odom is not None else ""
        if not frame or not odom or frame == odom:
            return points
        try:
            t = self.tf.lookup_transform(odom, frame, Time())
        except Exception as e:  # noqa: BLE001 (tf2's lookup, connectivity and extrapolation errors)
            self.get_logger().warning(f"no transform {frame} -> {odom}: {e}")
            return None
        yaw = _quat_to_rpy(t.transform.rotation)[2]
        c, s = math.cos(yaw), math.sin(yaw)
        return points @ np.array([[c, s], [-s, c]]) + np.array([t.transform.translation.x, t.transform.translation.y])

    def on_goal(self, msg: PoseStamped):
        self.get_logger().info(f"goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) in {msg.header.frame_id}")
        if self.planner is not None:  # stand until the planner's path arrives
            self.goal_msg = msg
            self.nav.clear()
            self.replan()
            return
        goal = self.to_odom(msg.header.frame_id, np.array([[msg.pose.position.x, msg.pose.position.y]]))
        if goal is not None:
            self.nav.set_goal(*goal[0])

    def on_path(self, msg: NavPath):
        if not msg.poses:
            return
        points = self.to_odom(msg.header.frame_id, np.array([[q.pose.position.x, q.pose.position.y] for q in msg.poses]))
        if points is not None:
            self.nav.set_path(points)

    def replan(self):
        """Ask the planner for a path from where the robot is to the goal."""
        if self.goal_msg is None or self.planning or not self.planner.server_is_ready():
            return
        from nav2_msgs.action import ComputePathToPose

        request = ComputePathToPose.Goal()
        request.goal = PoseStamped(pose=self.goal_msg.pose)
        request.goal.header.frame_id = self.goal_msg.header.frame_id  # stamp 0: the latest transforms
        request.use_start = False  # from the robot's pose in the planner's frame (TF)
        self.planning = True
        self.planner.send_goal_async(request).add_done_callback(self.on_plan_accepted)

    def on_plan_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.planning = False
            return
        handle.get_result_async().add_done_callback(self.on_plan)

    def on_plan(self, future):
        self.planning = False
        path = future.result().result.path
        if self.goal_msg is not None and path.poses:
            self.on_path(path)

    def pose(self) -> tuple[Pose, np.ndarray]:
        o = self.odom
        roll, pitch, yaw = _quat_to_rpy(o.pose.pose.orientation)
        pose = Pose(o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z, roll, pitch, yaw)
        t = o.twist.twist
        return pose, np.array([t.linear.x, t.linear.y, t.angular.z])

    def on_scan(self, msg: PointCloud2):
        self.scan_time = time.monotonic()
        if self.odom is None or self.mode == "sysid":
            return
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        pose, velocity = self.pose()
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        now = time.monotonic()
        depth = {name: pts for name, (t, pts) in self.depth.items() if now - t < self.timeout}
        command, info = self.nav.step(points, pose, velocity, stamp, depth)
        if info.get("state") == "reached" and self.goal_msg is not None:
            self.get_logger().info("goal reached")
            self.goal_msg = None  # no more replanning
        self.command_pub.publish(Float32MultiArray(data=[float(c) for c in command]))
        shield = info.get("shield", {})
        self.state_pub.publish(String(data=f"{info.get('state', '')} scale={shield.get('scale', 1.0):.2f} "
                                           f"gap={shield.get('min_gap', 1e3):.2f} person={shield.get('min_person_gap', 1e3):.2f}"))
        if self.mode == "drive" and self.out is not None:
            if self.stopped or not self.fresh():
                self.out.stop()
            else:
                self.out.send(*command)
        if self.record_path is not None:
            self.records.append({"stamp": stamp, "points": points.astype(np.float32),
                "depth": {k: v.astype(np.float32) for k, v in depth.items()}, "pose": np.array(
                [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]), "velocity": velocity, "command": command,
                "goal": self.nav.goal if self.nav.goal is not None else np.full(2, np.nan),
                "route": self.nav.route[0][0].numpy() if self.nav.route is not None else np.zeros((0, 2)),
                "route_version": self.nav.route_version, "max_speed": self.nav.max_speed})

    def fresh(self) -> bool:
        now = time.monotonic()
        return now - self.scan_time < self.timeout and now - self.odom_time < self.timeout

    def watchdog(self):
        if self.mode == "drive" and self.out is not None and (self.stopped or not self.fresh()):
            self.out.stop()

    # -- system identification ----------------------------------------------------------------------------------------

    def start_sysid(self, profile: str, log: str):
        rows = list(csv.DictReader(open(profile)))
        self.profile = np.array([[float(r["t"]), float(r["vx"]), float(r["vy"]), float(r["wz"])] for r in rows])
        self.sysid_log = open(log, "w", newline="")
        self.sysid_writer = csv.writer(self.sysid_log)
        self.sysid_writer.writerow(["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "x", "y", "yaw"])
        self.sysid_start = time.monotonic()
        self.create_timer(0.02, self.sysid_tick)
        self.get_logger().info(f"sysid: {len(rows)} commands over {self.profile[-1, 0]:.0f} s -> {log}")

    def sysid_tick(self):
        t = time.monotonic() - self.sysid_start
        if t > self.profile[-1, 0]:
            if self.out is not None:
                self.out.stop()
            self.sysid_log.close()
            self.get_logger().info("sysid done")
            rclpy.shutdown()
            return
        k = int(np.searchsorted(self.profile[:, 0], t, side="right")) - 1
        cmd = self.profile[max(k, 0), 1:]
        if self.out is not None and self.fresh_odom():
            self.out.send(*cmd)
        elif self.out is not None:
            self.out.stop()
        if self.odom is not None:
            pose, velocity = self.pose()
            self.sysid_writer.writerow([f"{t:.4f}", *(f"{c:.4f}" for c in cmd), *(f"{v:.4f}" for v in velocity),
                                        f"{pose.x:.4f}", f"{pose.y:.4f}", f"{pose.yaw:.4f}"])

    def fresh_odom(self) -> bool:
        return time.monotonic() - self.odom_time < self.timeout

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

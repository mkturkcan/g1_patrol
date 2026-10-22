"""ROS 2 node that drives the Unitree B2-W with the navigation policy (runs on the robot's Jetson Orin NX).

Subscribes to the lidar (``sensor_msgs/PointCloud2``, e.g. ``/rslidar_points`` from RoboSense's ``rslidar_sdk``),
odometry (``nav_msgs/Odometry``: base_link in the odometry frame, twist in base_link) and a goal
(``geometry_msgs/PoseStamped`` in the odometry frame, e.g. RViz's "2D Goal Pose"). One policy step per lidar scan; the
command goes to the robot's sport-mode controller, either through ``unitree_sdk2py``'s sport client
(``--ros-args -p output:=sdk``) or as ``geometry_msgs/Twist`` on ``cmd_vel`` for a bridge (``output:=twist``).

Modes (parameter ``mode``):

* ``shadow`` (default): compute and publish the commands the policy would give on ``~/command`` (and log them), but
  do not move the robot -- drive it by hand and watch what the policy would do;
* ``drive``: send the commands, capped at ``max_speed`` (default 0.5 m/s), with the lidar guard on; the command is
  zeroed if lidar or odometry are older than ``timeout`` s;
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
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, String

from .navigator import Navigator, OnnxPolicy, Pose


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

    def on_goal(self, msg: PoseStamped):
        self.nav.set_goal(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

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
        command, info = self.nav.step(points, pose, velocity, stamp)
        self.command_pub.publish(Float32MultiArray(data=[float(c) for c in command]))
        self.state_pub.publish(String(data=info.get("state", "")))
        if self.mode == "drive" and self.out is not None and self.fresh():
            self.out.send(*command)
        if self.record_path is not None:
            self.records.append({"stamp": stamp, "points": points.astype(np.float32), "pose": np.array(
                [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]), "velocity": velocity, "command": command,
                "goal": self.nav.goal if self.nav.goal is not None else np.full(2, np.nan)})

    def fresh(self) -> bool:
        now = time.monotonic()
        return now - self.scan_time < self.timeout and now - self.odom_time < self.timeout

    def watchdog(self):
        if self.mode == "drive" and self.out is not None and not self.fresh():
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
            np.savez_compressed(self.record_path, **{k: np.array([r[k] for r in self.records], dtype=object if k == "points" else None) for k in keys})
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

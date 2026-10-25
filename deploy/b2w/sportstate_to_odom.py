"""Odometry for the navigation stack from Unitree's sport-mode state: ``nav_msgs/Odometry`` and the TF odom -> base_link.

The sport-mode controller publishes its state estimate (leg and wheel odometry fused with the IMU) as
``unitree_go/msg/SportModeState`` on ``/sportmodestate`` (``unitree_ros2``): ``position`` (x, y, z in its odometry
frame), ``imu_state.quaternion`` (w, x, y, z) and ``imu_state.gyroscope`` (body rates). The pose is republished as is.
The twist the policy needs (forward, lateral, yaw rate in base_link) is computed from the pose itself -- the change of
position over the last ``window`` seconds rotated into the base frame, and the gyroscope's yaw rate -- so that it does
not depend on the frame of the state's ``velocity`` field, which differs between firmware versions.

Usage (started by ``launch/b2w_navigation.launch.py``):
    python3 -m b2w.sportstate_to_odom --ros-args -p state_topic:=/sportmodestate
"""

from __future__ import annotations

import math
from collections import deque

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster
from unitree_go.msg import SportModeState
from .params import declare


class Differentiator:
    """Planar velocity (world frame) from timed positions: the change over the last ``window`` seconds."""

    def __init__(self, window: float = 0.1):
        self.window = window
        self.history: deque = deque()

    def update(self, t: float, x: float, y: float) -> tuple[float, float] | None:
        self.history.append((t, x, y))
        while len(self.history) > 2 and t - self.history[1][0] >= self.window:
            self.history.popleft()
        t0, x0, y0 = self.history[0]
        if t - t0 < 0.3 * self.window:
            return None
        return (x - x0) / (t - t0), (y - y0) / (t - t0)


class SportStateToOdom(Node):
    def __init__(self):
        super().__init__("b2w_odometry")
        p = lambda name, default: declare(self, name, default)  # noqa: E731
        self.frame = p("odom_frame", "odom")
        self.base = p("base_frame", "base_link")
        self.period = 1.0 / float(p("rate", 50.0))
        self.diff = Differentiator(float(p("window", 0.1)))
        self.pub = self.create_publisher(Odometry, p("odom_topic", "/odom"), 20)
        self.tf = TransformBroadcaster(self) if p("publish_tf", True) else None
        self.create_subscription(SportModeState, p("state_topic", "/sportmodestate"), self.on_state,
                                 qos_profile_sensor_data)
        self.velocity = (0.0, 0.0)
        self.last_publish = -1.0
        self.count = 0
        self.create_timer(5.0, self.report)

    def on_state(self, s: SportModeState):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        x, y, z = (float(v) for v in s.position)
        w, qx, qy, qz = (float(v) for v in s.imu_state.quaternion)
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        v = self.diff.update(t, x, y)
        if v is not None:
            self.velocity = v
        self.count += 1
        if t - self.last_publish < self.period:
            return
        self.last_publish = t
        c, sn = math.cos(yaw), math.sin(yaw)
        vx_w, vy_w = self.velocity
        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id, msg.child_frame_id = self.frame, self.base
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = x, y, z
        o = msg.pose.pose.orientation
        o.w, o.x, o.y, o.z = w, qx, qy, qz
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = c * vx_w + sn * vy_w, -sn * vx_w + c * vy_w
        msg.twist.twist.angular.z = float(s.imu_state.gyroscope[2])
        self.pub.publish(msg)
        if self.tf is not None:
            tf = TransformStamped()
            tf.header = msg.header
            tf.child_frame_id = self.base
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
            tf.transform.rotation = msg.pose.pose.orientation
            self.tf.sendTransform(tf)

    def report(self):
        if self.count == 0:
            self.get_logger().warning("no sport-mode state received yet: is the robot on, the network set (CYCLONEDDS_URI)"
                                      " and unitree_ros2 sourced?")
        self.count = 0


def main():
    rclpy.init()
    node = SportStateToOdom()
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

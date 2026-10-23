"""Republish Unitree's sport-mode state as ``nav_msgs/Odometry`` for the navigation node.

The sport-mode controller publishes its state estimate (leg and wheel odometry fused with the IMU) as
``unitree_go/msg/SportModeState`` (``unitree_ros2``): ``position`` (x, y, z), ``imu_state.quaternion`` (w, x, y, z),
``velocity`` (vx, vy, vz) and ``yaw_speed``. The navigation node expects the pose of base_link in the odometry frame
and the twist in base_link. Whether ``velocity`` is expressed in the odometry or the body frame differs between
firmware versions: check it once (drive straight ahead after turning by 90 degrees; in the body frame vx stays
positive) and set ``velocity_frame`` accordingly. The pose is also broadcast as the TF odom -> base_link
(``publish_tf``), which a localization (AMCL) and Nav2's planner need.

Usage:
    python3 -m b2w.sportstate_to_odom --ros-args -p state_topic:=/sportmodestate -p velocity_frame:=world
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from unitree_go.msg import SportModeState


class SportStateToOdom(Node):
    def __init__(self):
        super().__init__("sportstate_to_odom")
        self.velocity_frame = self.declare_parameter("velocity_frame", "world").value
        self.frame = self.declare_parameter("odom_frame", "odom").value
        self.pub = self.create_publisher(Odometry, self.declare_parameter("odom_topic", "/odom").value, 20)
        self.tf = TransformBroadcaster(self) if self.declare_parameter("publish_tf", True).value else None
        self.create_subscription(SportModeState, self.declare_parameter("state_topic", "/sportmodestate").value, self.on_state, 20)

    def on_state(self, s: SportModeState):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id, msg.child_frame_id = self.frame, "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = (float(v) for v in s.position)
        w, x, y, z = (float(v) for v in s.imu_state.quaternion)
        o = msg.pose.pose.orientation
        o.w, o.x, o.y, o.z = w, x, y, z
        vx, vy = float(s.velocity[0]), float(s.velocity[1])
        if self.velocity_frame == "world":  # rotate into the base frame (yaw only: the robot stays level)
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, sn = math.cos(yaw), math.sin(yaw)
            vx, vy = c * vx + sn * vy, -sn * vx + c * vy
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = vx, vy
        msg.twist.twist.angular.z = float(s.yaw_speed)
        self.pub.publish(msg)
        if self.tf is not None:
            t = TransformStamped()
            t.header = msg.header
            t.child_frame_id = msg.child_frame_id
            p = msg.pose.pose.position
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
            t.transform.rotation = msg.pose.pose.orientation
            self.tf.sendTransform(t)


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

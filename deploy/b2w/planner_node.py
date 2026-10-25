"""ROS 2 global planner for the B2-W: routes on the map that SLAM builds while the robot drives.

Subscribes to the occupancy grid (``/map`` from slam_toolbox), a goal (``/goal_pose``, e.g. RViz's "2D Goal Pose", in
any frame TF knows) or a goal relative to the robot (``/b2w/goal_relative``: ``geometry_msgs/PointStamped``, x
forward and y left of base_link -- "drive 5 m ahead" without RViz), and finds the robot in the map with TF
(map -> base_link). Once a second while a goal is set, it plans a route (:mod:`global_planner`, the planner of the
training routes; unexplored space counts as free) and publishes it as ``nav_msgs/Path`` on ``/plan`` in the map
frame; the navigation node follows it. An empty path means: no route (the robot stands). The goal is cleared when
the robot is within ``success_threshold`` of it.

Usage (started by ``launch/b2w_navigation.launch.py``):
    python3 -m b2w.planner_node
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .global_planner import plan_on_grid
from .params import declare


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PlannerNode(Node):
    def __init__(self):
        super().__init__("b2w_planner")
        p = lambda name, default: declare(self, name, default)  # noqa: E731
        self.map_frame = p("map_frame", "map")
        self.base_frame = p("base_frame", "base_link")
        self.half_width = float(p("half_width", 0.29))
        self.resolution = float(p("resolution", 0.1))
        self.success = float(p("success_threshold", 0.5))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, p("map_topic", "/map"), self.on_map, latched)
        self.create_subscription(PoseStamped, p("goal_topic", "/goal_pose"), self.on_goal, 10)
        self.create_subscription(PointStamped, p("relative_goal_topic", "/b2w/goal_relative"), self.on_relative_goal, 10)
        self.path_pub = self.create_publisher(Path, p("path_topic", "/plan"), 10)
        self.state_pub = self.create_publisher(String, "/b2w/planner_state", 10)
        self.map: OccupancyGrid | None = None
        self.goal: np.ndarray | None = None
        self.busy = False
        self.create_timer(float(p("period", 1.0)), self.tick)
        self.get_logger().info("B2-W planner: waiting for the map and a goal (RViz '2D Goal Pose' or /b2w/goal_relative)")

    # -- inputs -------------------------------------------------------------------------------------------------------

    def on_map(self, msg: OccupancyGrid):
        self.map = msg

    def robot(self) -> tuple[float, float, float] | None:
        """The robot's pose (x, y, yaw) in the map frame, or None."""
        try:
            t = self.tf.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:  # noqa: BLE001 (tf2's lookup, connectivity and extrapolation errors)
            return None
        return t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation)

    def on_goal(self, msg: PoseStamped):
        frame = msg.header.frame_id or self.map_frame
        x, y = msg.pose.position.x, msg.pose.position.y
        if frame != self.map_frame:
            try:
                t = self.tf.lookup_transform(self.map_frame, frame, Time())
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"goal in {frame}, no transform to {self.map_frame}: {e}")
                return
            c, s = math.cos(_yaw(t.transform.rotation)), math.sin(_yaw(t.transform.rotation))
            x, y = t.transform.translation.x + c * x - s * y, t.transform.translation.y + s * x + c * y
        self.set_goal(x, y)

    def on_relative_goal(self, msg: PointStamped):
        pose = self.robot()
        if pose is None:
            self.get_logger().error("relative goal: the robot's pose in the map is not known yet (is SLAM running?)")
            return
        rx, ry, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        self.set_goal(rx + c * msg.point.x - s * msg.point.y, ry + s * msg.point.x + c * msg.point.y)

    def set_goal(self, x: float, y: float):
        self.goal = np.array([x, y])
        self.get_logger().info(f"goal ({x:.2f}, {y:.2f}) in {self.map_frame}")
        self.tick()

    # -- planning -----------------------------------------------------------------------------------------------------

    def tick(self):
        if self.goal is None or self.busy:
            return
        pose = self.robot()
        if pose is None or self.map is None:
            self.state("waiting for " + ("the map" if self.map is None else "the robot's pose (TF map -> base_link)"))
            return
        start = np.array(pose[:2])
        if float(np.hypot(*(self.goal - start))) < self.success:
            self.get_logger().info("goal reached")
            self.goal = None
            self.publish([])
            self.state("goal reached")
            return
        m = self.map
        grid = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        origin = (m.info.origin.position.x, m.info.origin.position.y)
        if abs(_yaw(m.info.origin.orientation)) > 1e-3:
            self.get_logger().error("rotated map origins are not supported")
            return
        self.busy = True
        goal = self.goal.copy()
        threading.Thread(target=self.plan, args=(grid, origin, m.info.resolution, start, goal), daemon=True).start()

    def plan(self, grid, origin, res, start, goal):
        try:
            t0 = time.perf_counter()
            route = plan_on_grid(grid, origin, res, start, goal, self.half_width, self.resolution)
            took = time.perf_counter() - t0
            if self.goal is None or not np.allclose(goal, self.goal):
                return  # the goal changed meanwhile
            if route is None:
                self.publish([])
                self.state(f"no route to ({goal[0]:.1f}, {goal[1]:.1f})")
                self.get_logger().warning("no route to the goal on the current map", throttle_duration_sec=5.0)
            else:
                self.publish(route)
                length = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(route[:-1], route[1:]))
                self.state(f"route {length:.1f} m, {len(route)} points, planned in {took * 1000:.0f} ms")
        finally:
            self.busy = False

    def publish(self, route: list):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for k, (x, y) in enumerate(route):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            nx, ny = route[min(k + 1, len(route) - 1)]
            yaw = math.atan2(ny - y, nx - x) if k + 1 < len(route) else 0.0
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def state(self, text: str):
        self.state_pub.publish(String(data=text))


def main():
    rclpy.init()
    node = PlannerNode()
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

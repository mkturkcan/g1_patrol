"""Software-in-the-loop check of the navigation stack with the simulated B2-W (``fake_b2w.py``).

Waits until SLAM localizes the robot (TF map -> base_link) and the navigation node reports, sends a goal (map frame,
or relative to the robot), then follows the simulated robot's true pose until it is within ``reach`` of the goal or
``timeout`` s pass. Prints a summary and exits with 0 if the goal was reached and the robot never moved into anything (people walking into
the stopped robot are counted, but do not fail the check unless ``allow_walk_ins:=false``).

Usage (see ``run_sil.sh``):
    python3 -m b2w.sim.sil_check --ros-args -p goal:="[4.0, 3.5]" -p timeout:=120.0
"""

from __future__ import annotations

import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Int32, String
from tf2_ros import Buffer, TransformListener
from ..params import declare


class SilCheck(Node):
    def __init__(self):
        super().__init__("b2w_sil_check")
        p = lambda name, default: declare(self, name, default)  # noqa: E731
        self.goal = [float(v) for v in p("goal", [4.0, 3.5])]
        self.timeout = float(p("timeout", 120.0))
        self.reach = float(p("reach", 0.6))
        self.trace_file = p("trace", "/tmp/b2w_sil_trace.json")
        self.allow_walk_ins = bool(p("allow_walk_ins", True))
        """Contacts of people walking into the (nearly) stopped robot do not fail the check."""
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 1)
        self.create_subscription(PoseStamped, "/fake_b2w/pose", self.on_pose, 10)
        self.create_subscription(Int32, "/fake_b2w/contacts", self.on_contacts, 10)
        self.create_subscription(Int32, "/fake_b2w/robot_moving_contacts", self.on_moving_contacts, 10)
        self.moving_contacts = 0
        self.create_subscription(String, "/b2w_nav/state", self.on_state, 10)
        self.create_subscription(String, "/b2w/planner_state", self.on_planner, 10)
        self.pose = None
        self.contacts = 0
        self.nav_state, self.planner_state = None, None
        self.sent = None
        self.path_length = 0.0
        self.max_speed = 0.0
        self.start = time.monotonic()
        self.last = None
        self.done = None
        self.create_timer(0.5, self.tick)
        self.trace = []

    def on_pose(self, msg):
        p = (msg.pose.position.x, msg.pose.position.y)
        if self.pose is not None and self.sent is not None:
            self.path_length += math.hypot(p[0] - self.pose[0], p[1] - self.pose[1])
        self.pose = p

    def on_contacts(self, msg):
        self.contacts = msg.data

    def on_moving_contacts(self, msg):
        self.moving_contacts = msg.data

    def on_state(self, msg):
        self.nav_state = json.loads(msg.data)
        if self.sent is not None and self.pose is not None:
            self.trace.append((round(time.monotonic() - self.sent, 2), *(round(v, 2) for v in self.pose),
                               self.nav_state.get("state"), self.nav_state.get("command"),
                               self.nav_state.get("moving_gap"), self.nav_state.get("crawl"), self.contacts))

    def on_planner(self, msg):
        self.planner_state = msg.data

    def tick(self):
        now = time.monotonic()
        if self.sent is None:
            try:
                self.tf.lookup_transform("map", "base_link", Time())
                ready = self.nav_state is not None
            except Exception:  # noqa: BLE001
                ready = False
            if ready and now - self.start > 5.0:
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.pose.position.x, goal.pose.position.y = self.goal
                goal.pose.orientation.w = 1.0
                self.goal_pub.publish(goal)
                self.sent = now
                self.get_logger().info(f"goal {self.goal} sent")
            elif now - self.start > 60.0:
                self.finish(False, "the stack did not come up (no TF map -> base_link or no navigation state)")
            return
        if self.pose is not None and math.hypot(self.pose[0] - self.goal[0], self.pose[1] - self.goal[1]) < self.reach:
            self.finish(self.moving_contacts == 0 and (self.contacts == 0 or self.allow_walk_ins), "goal reached")
        elif now - self.sent > self.timeout:
            self.finish(False, "timeout")
        elif self.nav_state is not None and now - (self.last or 0) > 5.0:
            self.last = now
            self.get_logger().info(f"t {now - self.sent:5.1f} s pose {tuple(round(v, 2) for v in self.pose or ())} "
                                   f"nav {self.nav_state.get('state')} cmd {self.nav_state.get('command')} "
                                   f"policy {self.nav_state.get('policy_command')} yaw {self.nav_state.get('yaw')} "
                                   f"gap {self.nav_state.get('min_gap')} "
                                   f"floor {self.nav_state.get('floor')} | planner: {self.planner_state}")

    def finish(self, ok: bool, why: str):
        if self.done is not None:
            return
        self.done = ok
        took = time.monotonic() - (self.sent or self.start)
        summary = {"ok": ok, "why": why, "seconds": round(took, 1), "driven_m": round(self.path_length, 2),
                   "contacts": self.contacts, "robot_moving_contacts": self.moving_contacts,
                   "final_pose": self.pose, "goal": self.goal,
                   "last_nav_state": self.nav_state, "last_planner_state": self.planner_state}
        print("SIL RESULT " + json.dumps(summary), flush=True)
        with open(self.trace_file, "w") as f:
            json.dump(self.trace, f)
        raise SystemExit(0 if ok else 1)


def main():
    rclpy.init()
    node = SilCheck()
    code = 1
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = int(e.code)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()

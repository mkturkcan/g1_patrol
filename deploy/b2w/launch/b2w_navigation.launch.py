"""Launch the B2-W navigation stack on the robot's Jetson (ROS 2 Foxy or Humble).

Starts:

* the static transforms of the lidar (and the depth cameras) on base_link, from the mounts in ``deploy.json``;
* the RoboSense lidar driver (``rslidar_sdk``, ``config/rslidar_helios.yaml``) unless ``lidar_driver:=false``;
* the two RealSense D435i (``realsense2_camera``) if ``cameras:=true`` (``front_serial``, ``rear_serial``);
* the odometry bridge (``b2w.sportstate_to_odom``: /sportmodestate -> /odom and odom -> base_link);
* SLAM (slam_toolbox on a 2D slice of the lidar: map and map -> odom) and the global planner (``b2w.planner_node``:
  goals from RViz or /b2w/goal_relative -> /plan), unless ``slam:=false``;
* the navigation node (``b2w.ros2_node``: the policy, the shield, commands to the sport-mode controller);
* RViz if ``rviz:=true`` (on a laptop on the robot's network, run ``rviz2 -d deploy/b2w/rviz/b2w.rviz`` instead).

Usage, from the repository root with ROS 2 and unitree_ros2 sourced (``deploy/b2w/README.md``)::

    ros2 launch deploy/b2w/launch/b2w_navigation.launch.py mode:=shadow
    ros2 launch deploy/b2w/launch/b2w_navigation.launch.py mode:=drive max_speed:=0.5
"""

import json
import math
import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "deploy" / "b2w"

ARGS = {
    "mode": ("shadow", "shadow: compute and show commands, send nothing; drive: send them (hold the dead-man button)"),
    "max_speed": ("0.5", "cap on the planar speed (m/s), 0.4-1.5; the policy is told the cap"),
    "release": (str(DEPLOY / "release"), "directory with policy.onnx and deploy.json"),
    "deadman": ("L1", "remote button that must be held for the policy to drive (or 'none')"),
    "output": ("ros", "ros: /api/sport/request (unitree_ros2); sdk: unitree_sdk2py; twist: /cmd_vel"),
    "network_interface": ("eth0", "network interface to the robot (output:=sdk only)"),
    "state_topic": ("/sportmodestate", "the sport-mode state (unitree_go/SportModeState)"),
    "lidar_driver": ("true", "start rslidar_sdk (false if the lidar's point cloud is published already)"),
    "lidar_topic": ("/rslidar_points", "the lidar's point cloud"),
    "lidar_frame": ("rslidar", "frame of the lidar's point cloud"),
    "cameras": ("false", "start the front and rear RealSense D435i (needs their serial numbers)"),
    "front_serial": ("", "serial number of the front D435i (rs-enumerate-devices)"),
    "rear_serial": ("", "serial number of the rear D435i"),
    "slam": ("true", "build the map while driving (slam_toolbox) and plan routes on it"),
    "rviz": ("false", "start RViz here"),
    "record": ("", "file (.npz) to record every scan, pose and command to (for scripts/b2w/replay.py)"),
    "device": ("cpu", "torch device for the map and the shield (cpu or cuda)"),
    "crawl": ("true", "crawl at 0.1 m/s within 1.2 m of anything that moves or moved in the last 2 s (false: faster)"),
}


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _ypr(m):
    """yaw, pitch, roll of a rotation matrix (static_transform_publisher's order)."""
    pitch = math.asin(max(-1.0, min(1.0, -m[2][0])))
    return math.atan2(m[1][0], m[0][0]), pitch, math.atan2(m[2][1], m[2][2])


def _static(x, y, z, yaw, pitch, roll, parent, child):
    args = [f"{v:.6f}" for v in (x, y, z, yaw, pitch, roll)] + [parent, child]
    return Node(package="tf2_ros", executable="static_transform_publisher", name=f"tf_{child}", arguments=args)


def _python(module: str, params: dict, name: str):
    """One of our nodes (plain Python modules, no ROS package needed)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "deploy"), env.get("PYTHONPATH", "")])
    ros_args = [a for k, v in params.items() for a in ("-p", f"{k}:={v}")]
    return ExecuteProcess(cmd=[sys.executable, "-m", module, "--ros-args", "-r", f"__node:={name}", *ros_args],
                          env=env, output="screen", name=name)


def _setup(context):
    get = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    release = Path(get["release"]).expanduser().resolve()
    cfg = json.loads((release / "deploy.json").read_text())
    actions = [LogInfo(msg=f"B2-W navigation: {release}, mode {get['mode']}, max speed {get['max_speed']} m/s")]

    # lidar mount (deploy.json) as the TF base_link -> lidar frame, for SLAM's scan and RViz
    mount = cfg["lidar"]["mount"]
    actions.append(_static(*mount["pos"], *_ypr(_rpy_matrix(*mount.get("rpy", (0.0, 0.0, 0.0)))), "base_link",
                           get["lidar_frame"]))
    if get["lidar_driver"] == "true":
        actions.append(Node(package="rslidar_sdk", executable="rslidar_sdk_node", name="rslidar_sdk", output="screen",
                            parameters=[{"config_path": str(DEPLOY / "config" / "rslidar_helios.yaml")}]))

    # depth cameras: base_link -> <name>_camera_link (realsense2_camera adds camera_link -> depth optical frame, a
    # rotation of rpy (-pi/2, 0, -pi/2))
    link_to_optical = _rpy_matrix(-math.pi / 2, 0.0, -math.pi / 2)
    optical_to_link = [list(r) for r in zip(*link_to_optical)]
    nav_params = {}
    for name, c in cfg.get("depth_cameras", {}).items():
        base_to_link = _matmul(_rpy_matrix(*c["rpy"]), optical_to_link)
        actions.append(_static(*c["pos"], *_ypr(base_to_link), "base_link", f"{name}_camera_link"))
        nav_params[f"{name}_depth_topic"] = f"/{name}_camera/depth/image_rect_raw"
        nav_params[f"{name}_info_topic"] = f"/{name}_camera/depth/camera_info"
        serial = get[f"{name}_serial"] if f"{name}_serial" in get else ""
        if get["cameras"] == "true":
            if not serial:
                actions.append(LogInfo(msg=f"cameras:=true but no {name}_serial given: the {name} camera is not started"))
                continue
            actions.append(Node(package="realsense2_camera", executable="realsense2_camera_node", name=f"{name}_camera",
                                namespace="", output="screen", parameters=[{
                                    "serial_no": f"_{serial}", "camera_name": f"{name}_camera",
                                    "enable_color": False, "enable_depth": True, "enable_infra1": False,
                                    "enable_infra2": False, "enable_gyro": False, "enable_accel": False,
                                    "depth_module.profile": "424x240x15", "publish_tf": True}]))

    # odometry
    actions.append(_python("b2w.sportstate_to_odom", {"state_topic": get["state_topic"]}, "b2w_odometry"))

    # SLAM on a slice of the lidar, and the global planner on its map
    if get["slam"] == "true":
        actions.append(Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
                            name="lidar_to_scan", remappings=[("cloud_in", get["lidar_topic"]), ("scan", "/scan")],
                            parameters=[{"target_frame": "base_link", "transform_tolerance": 0.05,
                                         # 0.15-0.3 m above the floor (the body stands 0.55-0.72 m high) to 1.0 m
                                         "min_height": -0.4, "max_height": 0.45, "angle_min": -math.pi,
                                         "angle_max": math.pi, "angle_increment": math.radians(0.5),
                                         "scan_time": 0.1, "range_min": 0.7, "range_max": 30.0, "use_inf": True}]))
        actions.append(Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox",
                            output="screen", parameters=[str(DEPLOY / "config" / "slam_toolbox.yaml")]))
        actions.append(_python("b2w.planner_node", {"success_threshold": cfg["success_threshold"],
                                                    "half_width": cfg["shield"]["half_width"]}, "b2w_planner"))

    # the policy
    params = {"config": str(release / "deploy.json"), "policy": str(release / "policy.onnx"), "mode": get["mode"],
              "max_speed": get["max_speed"], "deadman": get["deadman"], "output": get["output"],
              "network_interface": get["network_interface"], "lidar_topic": get["lidar_topic"], "device": get["device"],
              "crawl": get["crawl"], **nav_params}
    if get["slam"] != "true":
        params["goal_topic"] = "/goal_pose"  # straight-line goals in the odometry frame
    if get["record"]:
        params["record"] = get["record"]
    actions.append(_python("b2w.ros2_node", params, "b2w_nav"))
    if get["rviz"] == "true":
        actions.append(Node(package="rviz2", executable="rviz2", name="rviz", arguments=["-d", str(DEPLOY / "rviz" / "b2w.rviz")]))
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
                             + [OpaqueFunction(function=_setup)])

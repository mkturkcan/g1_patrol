"""Unitree B2-W: geometry, sensors and the settings its navigation policy is trained and deployed with.

No simulator imports (only the standard library): the simulation configs (:mod:`g1_patrol.tasks.navigation.config.b2w`), the export script
(``scripts/b2w/export.py``) and the robot-side code (``deploy/b2w``) all read these values, so that the robot
computes its observations exactly as in training. Values marked "measure" should be checked on the robot.
"""

import math

BASE_HEIGHT = 0.65
"""Height (m) of base_link above the ground when the B2-W stands: Unitree's standing pose (thigh 0.67 rad, calf
-1.3 rad; 0.35 m thigh and calf, 0.1125 m wheels) puts it at 0.67 m, the body's top at the 0.758 m of the spec."""
BASE_HEIGHT_RANGE = (0.55, 0.72)
"""Training draws the standing height per episode from this interval (the controller's body height is adjustable and
not known exactly); on the robot the floor is measured in every scan (``deploy/b2w/floor.py``), not assumed."""
GROUND_ERROR = (0.03, 0.01)
"""Error (m) of the floor height the robot measures: a bias that stays for the episode (up to the first value) and
noise per scan (standard deviation, the second); the map's heights are taken from it in training."""
FOOTPRINT = (0.56, 0.29)
"""Half length and half width (m) of the footprint (Unitree: 1098 x 550 mm standing, plus a little margin)."""
CORNER_RADIUS = 0.12
"""Rounding of the footprint's corners (m); the wheels are 225 mm across."""
BODY = (0.86, 0.3, 0.15)
"""Box (m) standing in for the body (head to tail, URDF): it shades the lidar as the real body does; the lidar sits
0.1 m above its top, so the ground is hidden within about 0.7 m ahead, 1 m to the sides and 5 m behind."""
CROP_BOX = ((0.0, 0.0, -0.25), (0.62, 0.34, 0.55))
"""Center and half sizes (m, base frame) of the box in which returns are the robot itself and are dropped: the body,
head and tail, and the legs and wheels down to the floor (the lidar sees the rear wheels)."""
LIDAR_MOUNT = (0.34218, 0.0, 0.17851)
"""lidar_link in base_link (Unitree's b2w_description URDF), mounted upright (measure)."""
LIDARS = {
    # model: (channels, vertical field of view (deg), maximum range (m))
    "helios_5515": (32, (-55.0, 15.0), 30.0),
    "helios_1615": (32, (-16.0, 15.0), 30.0),
    "bpearl": (32, (0.0, 90.0), 30.0),
}
"""RoboSense lidars that Unitree fits (check the label on the robot's lidar)."""
LIDAR = "helios_5515"
"""The lidar the student is trained with: RoboSense RS-Helios-5515 (32 lines, 360 deg x 70 deg)."""
DEPTH_CAMERAS = {
    # name: position (m) and orientation (roll, pitch, yaw; rad; R = Rz Ry Rx) of the depth optical frame (z along the
    # optical axis, x right, y down) in base_link, from the URDF's f_dc_link and r_dc_link: 45 deg down, front and rear
    "front": ((0.42161, 0.025, 0.061851), (-2.3562, 0.0, -1.5708)),
    "rear": ((-0.42244, -0.025, 0.038026), (-2.3562, 0.0, 1.5708)),
}
"""Intel RealSense D435i depth cameras (measure the mounts)."""
D435I = {"hfov": 87.0, "vfov": 58.0, "width": 64, "height": 40, "min_range": 0.3, "max_range": 4.0}
"""D435i depth: field of view (deg), rays per frame in simulation (a 13x decimated 848 x 480 image on the robot),
the range (m) of returns used in the height map."""
COMMAND_RANGES = ((-0.8, 1.5), (-0.3, 0.3), (-1.2, 1.2))
"""(min, max) of the forward, lateral and yaw-rate commands (m/s, m/s, rad/s); policy outputs are squashed into them
with a tanh."""
CONTROLLER = {
    "delay_range": (0.04, 0.16),
    "gain_range": ((0.85, 1.1), (0.6, 1.05), (0.85, 1.1)),
    "deadband_max": (0.05, 0.05, 0.06),
    "tau_range": ((0.15, 0.45), (0.15, 0.5), (0.08, 0.3)),
    "accel_range": ((1.0, 3.0), (0.5, 2.0), (2.0, 6.0)),
    "drift_max": 0.05,
    "noise_max": (0.05, 0.03, 0.05),
    "noise_time": 0.3,
}
"""Ranges of the controller model (:mod:`g1_patrol.controller_model`) drawn per training episode: latency (s), gain,
deadband, time constant (s), acceleration bound (m/s^2, m/s^2, rad/s^2) per axis (forward, lateral, yaw rate), yaw
drift (rad/s per m/s), velocity noise (standard deviation) and its correlation time (s). Before system identification
of the robot (``scripts/b2w/sysid.py``) these are guesses for Unitree's sport mode."""
POLICY_DT = 0.1
"""Navigation step (s): one lidar scan per step."""
SUCCESS_THRESHOLD = 0.5
"""Distance (m) to the goal at which it counts as reached."""
SPEED_LIMIT_RANGE = (0.4, 1.5)
"""Caps (m/s) on the planar speed the policy is trained with, one per episode, which it observes: the robot's
``max_speed`` must lie in this interval."""
ROUTE_WAYPOINTS = (1.0, 2.5)
"""The policy steers by the points of its planned route (a global planner's path) this far (m) ahead of its place
along it (:mod:`g1_patrol.route`)."""
MAP = {
    "size": 8.0,
    "resolution": 0.1,
    "memory_s": 8.0,
    "max_range": 20.0,
    "z_range": (-0.6, 1.0),
    "unknown_value": -1.0,
    "drop_above": 1.0,
}
"""Lidar height map (:class:`g1_patrol.heightmap.HeightMap`): 8 x 8 m at 0.1 m around the robot, returns kept for 8 s,
heights -0.6..1 m relative to the ground; returns from above 1 m (what the robot passes under) are dropped."""


def shield_settings() -> dict:
    """Settings of the safety shield (:class:`g1_patrol.shield.ShieldCfg`) for the B2-W."""
    return {"half_length": FOOTPRINT[0], "half_width": FOOTPRINT[1], "corner_radius": CORNER_RADIUS,
            "ground_z": -BASE_HEIGHT}


def deploy_config(lidar: str = LIDAR, depth: bool = True) -> dict:
    """Settings for the robot-side code (``deploy.json``); ``depth``: the policy also uses the depth cameras."""
    channels, vfov, max_range = LIDARS[lidar]
    return {
        "robot": "unitree_b2w",
        "policy_dt": POLICY_DT,
        "success_threshold": SUCCESS_THRESHOLD,
        "route_waypoints": list(ROUTE_WAYPOINTS),
        "speed_limit_range": list(SPEED_LIMIT_RANGE),
        "command": {"ranges": [list(r) for r in COMMAND_RANGES], "squash": "tanh"},
        "observation": {
            "obs": [f"waypoint{i}_{c}" for i in range(len(ROUTE_WAYPOINTS)) for c in "xy"]
            + ["vx", "vy", "wz", "gravity_x", "gravity_y", "gravity_z", "last_action_vx", "last_action_vy",
               "last_action_wz", "speed_limit"],
            "map": "lidar_map (3, H, W): accumulated, current scan, previous scan",
        },
        "map": {**{k: list(v) if isinstance(v, tuple) else v for k, v in MAP.items()}, "base_height": BASE_HEIGHT,
                "crop_box": [list(CROP_BOX[0]), list(CROP_BOX[1])]},
        "lidar": {"model": lidar, "channels": channels, "vertical_fov": list(vfov), "max_range": max_range,
                  "mount": {"pos": list(LIDAR_MOUNT), "rpy": [0.0, 0.0, 0.0]},
                  # the simulated lidar casts channels x 360 rays (1 deg); denser real scans are thinned to this
                  "max_points": channels * 360},
        "shield": shield_settings(),
        "depth_cameras": {name: {"pos": list(pos), "rpy": list(rpy), **D435I} for name, (pos, rpy) in DEPTH_CAMERAS.items()}
        if depth else {},
    }


def rpy_matrix(rpy) -> list[list[float]]:
    """Rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll) (URDF convention)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def depth_camera_quat(name: str) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) in base_link of a depth camera's frame in the simulation's convention (x along the
    optical axis, y left, z up), from its optical frame (z along the axis, x right, y down) in :data:`DEPTH_CAMERAS`."""
    r = rpy_matrix(DEPTH_CAMERAS[name][1])
    # columns of the camera frame in base_link: x = optical z, y = -optical x, z = -optical y
    m = [[r[i][2], -r[i][0], -r[i][1]] for i in range(3)]
    w = math.sqrt(max(0.0, 1.0 + m[0][0] + m[1][1] + m[2][2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) / 2, m[2][1] - m[1][2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0][0] + m[1][1] - m[2][2])) / 2, m[0][2] - m[2][0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0][0] - m[1][1] + m[2][2])) / 2, m[1][0] - m[0][1])
    return (x, y, z, w)

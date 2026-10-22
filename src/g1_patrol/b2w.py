"""Unitree B2-W: geometry, sensors and the settings its navigation policy is trained and deployed with.

No simulator imports: the simulation configs (:mod:`g1_patrol.tasks.navigation.config.b2w`), the export script
(``scripts/b2w/export.py``) and the robot-side code (``deploy/b2w``) all read these values, so that the robot
computes its observations exactly as in training. Values marked "measure" should be checked on the robot.
"""

BASE_HEIGHT = 0.55
"""Height (m) of base_link above the ground when the B2-W stands (measure)."""
FOOTPRINT = (0.56, 0.29)
"""Half length and half width (m) of the footprint (Unitree: 1098 x 550 mm standing, plus a little margin)."""
CORNER_RADIUS = 0.12
"""Rounding of the footprint's corners (m); the wheels are 225 mm across."""
BODY = (0.86, 0.3, 0.15)
"""Box (m) standing in for the body (head to tail, URDF): it shades the lidar as the real body does; the lidar sits
0.1 m above its top, so the ground is hidden within about 0.7 m ahead, 1 m to the sides and 5 m behind."""
CROP_BOX = ((0.0, 0.0, -0.15), (0.6, 0.33, 0.45))
"""Center and half sizes (m, base frame) of the box in which lidar returns are the robot itself and are dropped."""
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
"""The lidar the student is trained with."""
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


def deploy_config(lidar: str = LIDAR) -> dict:
    """Settings for the robot-side code (``deploy.json``)."""
    channels, vfov, max_range = LIDARS[lidar]
    return {
        "robot": "unitree_b2w",
        "policy_dt": POLICY_DT,
        "success_threshold": SUCCESS_THRESHOLD,
        "command": {"ranges": [list(r) for r in COMMAND_RANGES], "squash": "tanh"},
        "observation": {
            "obs": ["goal_x", "goal_y", "vx", "vy", "wz", "gravity_x", "gravity_y", "gravity_z", "last_action_vx",
                    "last_action_vy", "last_action_wz"],
            "map": "lidar_map (3, H, W): accumulated, current scan, previous scan",
        },
        "map": {**{k: list(v) if isinstance(v, tuple) else v for k, v in MAP.items()}, "base_height": BASE_HEIGHT,
                "crop_box": [list(CROP_BOX[0]), list(CROP_BOX[1])]},
        "lidar": {"model": lidar, "channels": channels, "vertical_fov": list(vfov), "max_range": max_range,
                  "mount": {"pos": list(LIDAR_MOUNT), "rpy": [0.0, 0.0, 0.0]},
                  # the simulated lidar casts channels x 360 rays (1 deg); denser real scans are thinned to this
                  "max_points": channels * 360},
        "guard": {"half_length": FOOTPRINT[0], "half_width": FOOTPRINT[1], "horizon": 0.6, "min_reach": 0.25,
                  "min_points": 3},
    }

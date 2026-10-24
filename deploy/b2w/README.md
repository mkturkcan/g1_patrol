# Unitree B2-W: deployment and sim-to-real checks

The B2-W navigation policy commands body velocities to Unitree's sport-mode controller; it was trained against a
randomized model of that controller, a simulated RoboSense RS-Helios-5515 lidar and front and rear RealSense D435i depth
cameras. This note specifies the interface and a sequence of inexpensive checks that test the simulation's assumptions
on the robot before the policy drives it. Nothing here has been tested on hardware yet.

The released policies: `deploy/b2w/release` (lidar and depth cameras) and `deploy/b2w/release_lidar` (lidar only),
each with `policy.onnx` and its weights `policy.onnx.data`, `deploy.json` (the settings of the robot-side code) and
`samples.npz` (inputs and outputs from simulation, for `scripts/b2w/benchmark.py --check`); copy the whole directory.

## Interface

| Stage | Rate | Input | Output |
|---|---|---|---|
| Global planner (e.g. Nav2 `planner_server`) | per goal, then 1 Hz | map, localization, goal | path (`nav_msgs/Path`) |
| Route following (`g1_patrol.route`) | per scan | path, odometry pose | 2 waypoints ahead on the path |
| Height map (`g1_patrol.heightmap`) | per lidar scan (10 Hz) | scan (lidar frame), latest depth frames (optical frames), odometry pose | 3 × 80 × 80 map |
| Policy (`policy.onnx`) | per scan | `obs` (14), `height_map` (3 × 80 × 80) | 3 raw actions |
| Command shaping | per scan | raw actions | `(vx, vy, wz)` = `mid + half · tanh(a)`, planar speed capped at `max_speed` |
| Safety shield (`g1_patrol.shield`) | per scan | returns, map, measured velocity | the command, scaled down if needed |
| Sport-mode controller | Unitree | `Move(vx, vy, wz)` | legs and wheels |

Command ranges: `vx ∈ [−0.8, 1.5]` m/s, `vy ∈ [−0.3, 0.3]` m/s, `wz ∈ [−1.2, 1.2]` rad/s. In training every episode
caps the planar speed at a value between 0.4 and 1.5 m/s that the policy is given as an input; on the robot that cap is
the node's `max_speed` (keep it in that interval), so a slow first trial is a setting the policy knows, not a cut
applied behind its back. The values used in training
and on the robot come from one module, `src/g1_patrol/b2w.py`; `scripts/b2w/export.py` writes them to `deploy.json`.

`obs` (14 values, in order):

| Term | Size | Frame / units | Noise in training |
|---|---|---|---|
| route waypoints | 4 | the points of the route 1 m and 2.5 m ahead of the robot's place on it (the goal where it ends) − position, in the heading frame (x forward, y left), m | ±0.05; bias ±0.1 per episode |
| velocity | 3 | odometry, base frame: forward, lateral (m/s), yaw rate (rad/s) | ±0.05; bias ±0.03 per episode |
| projected gravity | 3 | gravity unit vector in the base frame (IMU) | ±0.03 |
| last action | 3 | the policy's previous raw output | – |
| speed limit | 1 | the cap on the planar speed, `max_speed` (m/s) | – |

`height_map` (3 × 80 × 80, 0.1 m cells, 8 × 8 m centred on the robot; rows: forward, columns: left): the highest return
of the lidar and the depth cameras per cell relative to the ground under the robot (base height − 0.55 m), clipped to
[−0.6, 1.0] m, −1 where unknown. The cameras look 45° down at the ground the lidar cannot see (within about 0.7 m in
front, 5 m behind); their returns count from 0.3 to 4 m, sampled on a 64 × 40 grid of the depth image. In training a
camera delivers nothing in 5 % of the episodes. A fallback policy uses the lidar alone (`release_lidar`, input
`lidar_map`).
Channel 0 accumulates scans for 8 s with odometry (the simulation drifts it by 5 mm and 2 mrad per step); channels 1
and 2 are the current and the previous scan. Returns inside the crop box around the robot (its own body) and returns
more than 1 m above the ground (what the robot passes under) are dropped; scans are thinned to the simulated lidar's
density (32 × 360 rays).

The robot's place on the route is the arc length of the closest point of the route, searched from 1 m behind to 3 m
ahead of the previous place (so that it does not skip to a part of the route on the other side of a wall); a new path
(the planner replanned) is searched whole once. In training the routes come from a planner on the tile's clearance map
that keeps up to 0.6 m from structures where that costs little detour (`src/g1_patrol/tasks/navigation/nyc/routing.py`);
in 30 % of the episodes the route is planned without small structures (chairs, carts, bins, posts), which the policy
has to get around itself, as with a map that is out of date. A goal given without a planner is reached in a straight
line, which is only right in open space. In a quarter of the episodes the robot starts facing a random way.

Footprint used in training: 1.12 × 0.58 m with rounded corners. Structures whose bottom is higher than 1 m do not
block the robot; drops (stairwells, a loading dock) end an episode.

## Safety shield

The shield runs on the NX after the policy and is independent of it; it can only slow the robot down along the path
it was taking, never steer it. A command is admissible if the robot, executing it, can still stop before anything it
sees comes within a margin: it keeps the commanded arc for 0.35 s (sensing, policy, controller latency), then brakes at
0.8 m/s² (allowing 12 % more speed than commanded and 8 cm of drift per meter); the footprint is swept along this
stopping path and the command is scaled down until it comes to rest in time. Margins: 0.10 m to obstacles, 0.25 m to
drops; to tall things (people, but also machines and walls) from 0.10 m below 0.25 m/s to 0.45 m above 0.75 m/s, so
doors are passed slowly and people with room. Returns where the current scan sees something tall and the previous one
saw floor, in a cell that held nothing tall for the last second, are taken to be people walking at the robot at
1.4 m/s while it closes in on them (the second excludes the edges of furniture, which flicker between floor and tall
from scan to scan as the lidar's rays fall differently). The closing speed towards a tall thing is held to 0.25 m/s
within 0.8 m. Moving away is never limited; if the robot's own momentum already breaches, the command is zero. A
person detector on the cameras' color images can mark people (`person`), who then keep 0.45 m at any speed.

The shield's deceleration and latency must be at least as pessimistic as the identified controller (`sysid.py fit`;
`scripts/b2w/evaluate.py --shield --shield_set latency=...,brake=...` evaluates other values). It does not see low
objects beside the robot that neither the lidar nor the cameras cover, except as remembered by the map. Tests
(`tests/test_shield.py`) cover doors, walls, drops, a person walking at the robot and closed loops through the slowest
controller of the training ranges. It does not replace the remote's stop: keep it in hand.

## Results in simulation

Evaluations of the release (`scripts/reproduce/b2w_3_evaluate.sh`; `scripts/b2w/results_table.py` prints the tables)
on the lab mixture: half the tiles are sections of the lab hall, half street and indoor tiles, robots spread over all
ten difficulty levels with up to 20 pedestrians, 1,024 robots for 300 s. Speed caps are drawn per episode from
0.4–1.5 m/s unless fixed; with the shield, the speed cap is applied before the shield, as on the robot. A pedestrian
contact counts as the robot's if it was moving towards the person at more than 0.1 m/s.

<!-- table:b2w -->
| Policy | Episodes | Goal (%) | Goal, lab hall (%) | Structure contact (%) | Pedestrian contact, robot moving (%) | Pedestrian walked into stopped robot (%) | Time-out (%) | Speed (m/s) | Shield stops (% of steps) |
|---|---|---|---|---|---|---|---|---|---|
| Lab teacher (privileged) | 19,239 | 68.6 | 78.7 | 14.3 | 11.5 | 3.9 | 1.3 | 0.54 | – |
| Fused student (lidar + 2 depth cameras) | 19,583 | 70.1 | 79.9 | 9.5 | 11.2 | 7.8 | 1.2 | 0.52 | – |
| Lidar-only student | 19,442 | 69.5 | 81.1 | 8.9 | 12.3 | 7.7 | 1.5 | 0.52 | – |
| Fused, shield | 13,322 | 73.7 | 83.2 | 2.7 | 1.5 | 14.5 | 7.5 | 0.37 | 11.5 |
| Fused, shield, cap 0.5 m/s | 11,685 | 70.2 | 81.7 | 2.4 | 2.2 | 17.4 | 7.8 | 0.31 | 9.5 |
| Fused, shield, cap 1.0 m/s | 14,502 | 74.8 | 83.8 | 2.5 | 1.3 | 14.5 | 6.8 | 0.40 | 12.8 |
<!-- /table:b2w -->

<!-- table:b2w_stress -->
| Fused student | Episodes | Goal (%) | Goal, lab hall (%) | Structure contact (%) | Pedestrian contact, robot moving (%) | Pedestrian walked into stopped robot (%) | Time-out (%) | Speed (m/s) | Shield stops (% of steps) |
|---|---|---|---|---|---|---|---|---|---|
| Nominal controller | 19,583 | 70.1 | 79.9 | 9.5 | 11.2 | 7.8 | 1.2 | 0.52 | – |
| Laggy controller (0.16-0.25 s latency) | 19,679 | 69.3 | 79.4 | 9.8 | 12.0 | 7.8 | 1.2 | 0.52 | – |
| Sluggish controller (2x response time, 1/2 acceleration) | 19,531 | 69.0 | 79.0 | 9.5 | 12.9 | 7.3 | 1.2 | 0.51 | – |
| Weak controller (70-80 % of the command) | 16,256 | 69.0 | 79.7 | 9.1 | 10.8 | 9.4 | 1.7 | 0.43 | – |
| Drifting controller (2x yaw drift) | 19,741 | 69.8 | 79.7 | 9.5 | 11.4 | 7.9 | 1.3 | 0.52 | – |
| Depth cameras failed | 19,168 | 70.1 | 80.2 | 9.7 | 10.8 | 7.9 | 1.5 | 0.51 | – |
| Degraded lidar map (20 % dropout, odometry drift) | 19,741 | 69.4 | 79.6 | 10.0 | 11.9 | 7.5 | 1.2 | 0.52 | – |
| Laggy controller, shield | 13,553 | 73.0 | 82.7 | 3.0 | 2.1 | 15.1 | 6.8 | 0.37 | 11.9 |
<!-- /table:b2w_stress -->

Most of the remaining pedestrian contacts with the shield are simulated pedestrians walking into a robot that has
stopped (on average the robot approached at 0.03 m/s, the pedestrian at 0.75 m/s); a quarter of the simulated
pedestrians (10 % at the easiest level, 40 % at the hardest) are distracted and give way only at the last moment.
With a perfect person detector (`evaluate.py --person_oracle`: people keep 0.45 m), contacts in which the robot was
moving fall from about 2 % to 0.55 % of the episodes at 0.5 m/s. The lidar-only student is as good as the fused one
in simulation; the cameras see the ground close around the robot that the lidar misses (low obstacles beside it),
which the simulated tiles rarely test.

## Values to check on the robot

| Value | Where | Assumed |
|---|---|---|
| Lidar model | `b2w.LIDAR`, `b2w.LIDARS` | RoboSense RS-Helios-5515: 32 lines, −55° to +15° (confirmed) |
| Lidar mount (x, y, z, rpy in base_link) | `b2w.LIDAR_MOUNT` | URDF `lidar_link`: (0.342, 0, 0.179), upright |
| Depth camera mounts (optical frames) | `b2w.DEPTH_CAMERAS` | URDF `f_dc_link`, `r_dc_link`: 45° down, front and rear |
| Depth topics (16-bit, mm) and camera info | node parameters | `/<front, rear>_camera/depth/image_rect_raw` |
| Base height above the floor, standing | `b2w.BASE_HEIGHT` | 0.55 m |
| Odometry: frame, rate, twist in base frame | ROS topic | `nav_msgs/Odometry`, ≥ 20 Hz |

A wrong base height or mount shows up as a floor that does not read 0 in the replayed maps (below).

## Checks, cheapest first

1. **Inference on the Jetson.** `scripts/b2w/benchmark.py --check samples.npz` runs the exported policy on inputs
   saved in simulation and compares the actions (catches a broken ONNX Runtime/TensorRT setup), then times a full
   step on a scan of the lidar's size. One step must fit well within the 100 ms scan period.
2. **Controller identification.** `scripts/b2w/sysid.py profile` writes a 2.5-minute command profile (steps, small
   commands, turns, arcs, sinusoids; it returns near its start). Play it in an open area with the node in
   `mode:=sysid`, then `sysid.py fit --log sysid_log.csv --json sysid.json` fits latency, gain, deadband, response
   time, acceleration bound and yaw drift, and flags values outside the training ranges (`b2w.CONTROLLER`).
   `scripts/b2w/evaluate.py --controller sysid:sysid.json` then evaluates the policy with the identified controller.
3. **Shadow drive.** Drive the robot by hand with the node in `mode:=shadow` and `record:=drive.npz`: it computes and
   publishes the policy's commands on `~/command` without sending them. Watch them in RViz or a plot: they should
   point along the path and slow down near people and in doorways.
4. **Map replay.** `scripts/b2w/replay.py --record drive.npz` rebuilds the maps offline, checks that it reproduces
   the node's commands and reports map statistics; `evaluate.py --map_stats` reports the same statistics in
   simulation. The floor should read about 0, and the share of known cells (overall, within 1.5 m) should be
   comparable.
5. **Twin evaluation.** Map the room (e.g. slam_toolbox with the lidar), draft a twin with
   `scripts/b2w/occupancy_to_twin.py`, add doors, tables and routes (see `lab_twin_example.yaml`), and run
   `evaluate.py --twin room.yaml` with the identified controller. Low success there predicts failure on the robot.
6. **Slow closed loop.** `mode:=drive` with `max_speed:=0.3`, the shield on, an operator on the remote's stop, the
   empty aisle first; then a person walking towards the robot (it must stop well short), then normal traffic. The
   robot stops if lidar or odometry stop for 0.3 s and while `~/stop` is true.

## The lab

The released policy is specialized on the lab it is deployed in: half its training tiles are sections of that hall
(`src/g1_patrol/tasks/navigation/nyc/makerlab.py`, modeled on photographs: the painted main aisle, drawer-cabinet
benches, test machines, open racks, bins, carts, pallets, a workstation area with office chairs and camera tripods;
thin-legged objects are kept out of as a whole), and its students see the polished concrete floor lose far returns. A
scan of the actual hall turned into a twin (step 5) adds its exact layout to the evaluation.

## Running the node

```bash
PYTHONPATH=<repo>/src:<repo>/deploy python3 -m b2w.ros2_node --ros-args \
    -p config:=deploy.json -p policy:=policy.onnx -p mode:=shadow \
    -p lidar_topic:=/rslidar_points -p odom_topic:=/odom -p goal_topic:=/goal_pose \
    -p planner_action:=/compute_path_to_pose -p record:=drive.npz
```

With `planner_action`, the node asks a planner for the path to each goal and again every second; Nav2's
`planner_server` with `map_server` and a localization (`amcl`, or slam_toolbox in localization mode) on a map of the
lab made beforehand with the lidar is enough. Nav2's `controller_server` and `bt_navigator` must not run: this node
takes their place. The path is brought into the odometry frame with TF (`map` -> `odom`) when it arrives. Without a
planner the node follows any `nav_msgs/Path` published on `path_topic` (default `/plan`), or a goal in a straight line.

The TF tree Nav2 needs: `map` -> `odom` from the localization, `odom` -> `base_link` from `b2w.sportstate_to_odom`
(or the robot's own odometry node), and the lidar's mount,
`ros2 run tf2_ros static_transform_publisher --x 0.342 --z 0.179 --frame-id base_link --child-frame-id rslidar`
(the frame of the Helios's point cloud; the policy node takes the mount from `deploy.json` instead). AMCL and
slam_toolbox take a laser scan: `pointcloud_to_laserscan` on `/rslidar_points`, target frame `base_link`, heights
−0.25 to 0.45 m (0.3 to 1.0 m above the floor).

Commands go to the sport-mode controller through `unitree_sdk2py` (`-p output:=sdk -p network_interface:=eth0`) or as
`geometry_msgs/Twist` on `cmd_vel` for an existing bridge (`-p output:=twist`). If the robot provides its state
estimate only as Unitree's `SportModeState`, `python3 -m b2w.sportstate_to_odom` republishes it as odometry (check
once in which frame its velocity is expressed; see the module). The robot-side code needs Python 3,
PyTorch, NumPy, ONNX Runtime and ROS 2 (rclpy, sensor_msgs_py, tf2_ros; nav2_msgs with a planner), not Isaac Lab.

# Unitree B2-W: deployment and sim-to-real checks

The B2-W navigation policy commands body velocities to Unitree's sport-mode controller; it was trained against a
randomized model of that controller, a simulated RoboSense RS-Helios-5515 lidar and front and rear RealSense D435i depth
cameras. This note specifies the interface and a sequence of inexpensive checks that test the simulation's assumptions
on the robot before the policy drives it. Nothing here has been tested on hardware yet.

## Interface

| Stage | Rate | Input | Output |
|---|---|---|---|
| Global planner (e.g. Nav2 `planner_server`) | per goal, then 1 Hz | map, localization, goal | path (`nav_msgs/Path`) |
| Route following (`g1_patrol.route`) | per scan | path, odometry pose | 2 waypoints ahead on the path |
| Height map (`g1_patrol.heightmap`) | per lidar scan (10 Hz) | scan (lidar frame), latest depth frames (optical frames), odometry pose | 3 × 80 × 80 map |
| Policy (`policy.onnx`) | per scan | `obs` (13), `height_map` (3 × 80 × 80) | 3 raw actions |
| Command shaping | per scan | raw actions | `(vx, vy, wz)` = `mid + half · tanh(a)` |
| Safety shield (`g1_patrol.shield`) | per scan | returns, map, measured velocity | the command, scaled down if needed |
| Sport-mode controller | Unitree | `Move(vx, vy, wz)` | legs and wheels |

Command ranges: `vx ∈ [−0.8, 1.5]` m/s, `vy ∈ [−0.3, 0.3]` m/s, `wz ∈ [−1.2, 1.2]` rad/s. The values used in training
and on the robot come from one module, `src/g1_patrol/b2w.py`; `scripts/b2w/export.py` writes them to `deploy.json`.

`obs` (13 values, in order):

| Term | Size | Frame / units | Noise in training |
|---|---|---|---|
| route waypoints | 4 | the points of the route 1 m and 2.5 m ahead of the robot's place on it (the goal where it ends) − position, in the heading frame (x forward, y left), m | ±0.05; bias ±0.1 per episode |
| velocity | 3 | odometry, base frame: forward, lateral (m/s), yaw rate (rad/s) | ±0.05; bias ±0.03 per episode |
| projected gravity | 3 | gravity unit vector in the base frame (IMU) | ±0.03 |
| last action | 3 | the policy's previous raw output | – |

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
doors are passed slowly and people with room; returns that moved since the previous scan are assumed to walk at the
robot at 1.4 m/s while it closes in on them. The closing speed towards a tall thing is held to 0.25 m/s within 0.8 m.
Moving away is never limited; if the robot's own momentum already breaches, the command is zero. A person detector on
the cameras' color images can mark people (`person`), who then keep 0.45 m at any speed.

In simulation (`scripts/b2w/evaluate.py --shield`, fed by the simulated lidar, depth cameras and map), driving at full
speed into everything for 30 s in 128 environments, the robot hit no structure and was still moving (more than
0.1 m/s) at one of 21 pedestrian contacts; the other 20 were pedestrians walking into the stopped robot. Tests
(`tests/test_shield.py`) cover doors, walls, drops, a person walking at the robot and closed loops through the slowest
controller of the training ranges. Its deceleration and latency must be at least as pessimistic as the identified
controller (`sysid.py fit`); the shield does not see low objects beside the robot that neither the lidar nor the cameras
cover, except as remembered by the map. It does not replace the remote's stop: keep it in hand.

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

Commands go to the sport-mode controller through `unitree_sdk2py` (`-p output:=sdk -p network_interface:=eth0`) or as
`geometry_msgs/Twist` on `cmd_vel` for an existing bridge (`-p output:=twist`). If the robot provides its state
estimate only as Unitree's `SportModeState`, `python3 -m b2w.sportstate_to_odom` republishes it as odometry (check
once in which frame its velocity is expressed; see the module). The robot-side code needs Python 3,
PyTorch, NumPy, ONNX Runtime and ROS 2 (rclpy, sensor_msgs_py, tf2_ros; nav2_msgs with a planner), not Isaac Lab.

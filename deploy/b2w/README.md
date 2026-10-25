# Unitree B2-W: running the navigation policy on the robot

This directory holds everything that runs on the B2-W's Jetson Orin NX: the navigation policy trained in simulation,
the safety shield, SLAM and a route planner, and one launch file that starts them. You install it once, check that
each part receives what it needs, map the lab while watching what the policy would do (it sends nothing), and then let
it drive while you hold a button on the remote. Nothing has to be measured on the robot: the floor under it is
measured in every scan, and the map is built while it drives.

Status: the whole stack has been run against a simulated B2-W on ROS 2 (below, "Trying it without the robot"); it has
not run on the robot yet.

## What runs where

| Where | What |
|---|---|
| B2-W (Unitree) | the sport-mode controller (legs and wheels); state on `/sportmodestate`, the remote on `/wirelesscontroller`, velocity requests on `/api/sport/request` |
| Jetson (expansion dock) | `launch/b2w_navigation.launch.py`: lidar driver, the D435i cameras (optional), odometry, SLAM, the route planner, the policy and the shield |
| Laptop (optional) | RViz: the map, the route, the robot; goals with "2D Goal Pose" |

Every lidar scan (10 Hz): SLAM updates the map from a slice of the scan; the planner (once a second) plans a route on
the map from the robot to the goal, through space not yet seen if needed; the policy gets the two points of the route
1 and 2.5 m ahead, its velocity, the speed cap and a height map built from the scan (and the cameras), and commands a
velocity; the shield lets through only what keeps the robot clear of everything it sees; the command goes to the
sport-mode controller as Move(vx, vy, wz). The policy drives only while the dead-man button (L1) is held.

## Step by step

### 1. Network

The Jetson is on the robot's internal network (192.168.123.x). Find its interface and address there:

```bash
ip -4 -o addr | grep 192.168.123.     # e.g. "eth0 ... 192.168.123.18/24"
```

A laptop for RViz joins the same network (the robot's Wi-Fi, or Ethernet with a static address such as
192.168.123.99/24) and uses the same `ROS_DOMAIN_ID` (default 0).

### 2. Install (once)

```bash
git clone https://github.com/mkturkcan/g1_patrol.git ~/g1_patrol && cd ~/g1_patrol
bash deploy/b2w/install.sh eth0          # the interface from step 1
source ~/b2w_env.sh                      # in every new shell
```

`install.sh` installs (with `sudo apt`) slam_toolbox, pointcloud_to_laserscan, CycloneDDS and realsense2_camera for
the ROS 2 of the Jetson's Ubuntu (Foxy on 20.04, Humble on 22.04; ROS 2 itself if missing), the Python packages
(numpy, scipy, onnxruntime, PyTorch), builds Unitree's message packages and RoboSense's lidar driver in `~/b2w_ws`, and
writes `~/b2w_env.sh` (ROS 2, the workspace, CycloneDDS on that interface). It ends by checking the policy on the
Jetson (`scripts/b2w/benchmark.py`: the same actions as in simulation, and the time per scan, which must stay well
under 100 ms).

### 3. Check the parts (with the robot on and standing)

```bash
ros2 topic hz /sportmodestate          # the robot's state: tens to hundreds of Hz
ros2 topic echo --once /wirelesscontroller   # the remote: hold L1 and look at "keys"
ros2 topic hz /rslidar_points          # the lidar, ~10 Hz -- only once the driver runs (step 4) or if it runs already
```

* No `/sportmodestate`: the interface in `~/b2w_env.sh` is not the robot's, or the robot is off.
* The lidar: if `ros2 topic list` already shows a point cloud from the robot's own software, run the stack with
  `lidar_driver:=false lidar_topic:=<that topic>`. Otherwise the stack starts RoboSense's driver, which receives the
  lidar's UDP packets on ports 6699/7788: `sudo tcpdump -i eth0 udp port 6699 -c 5` must show packets addressed to
  the Jetson. If they go elsewhere, set the lidar's destination address to the Jetson's in the lidar's web page.
* The D435i cameras are optional (the policy was trained with cameras that fail, and drives without them). If they are
  plugged into the Jetson, `rs-enumerate-devices -s` lists their serial numbers; cover one to see which is the front.

### 4. Shadow run: map the lab and watch the policy (it sends nothing)

```bash
ros2 launch deploy/b2w/launch/b2w_navigation.launch.py mode:=shadow
# with the cameras: ... cameras:=true front_serial:=<serial> rear_serial:=<serial>
```

On the laptop: `source ~/b2w_env.sh` (written by `install.sh` there too, or ROS 2 with the same CycloneDDS settings)
and `rviz2 -d deploy/b2w/rviz/b2w.rviz`. Drive the robot through the lab with the remote: the map grows. Set a goal
with "2D Goal Pose": the planned route appears (magenta), and the policy's two waypoints and the footprint are drawn
on the robot. `ros2 topic echo /b2w_nav/state` shows what the policy would do (`policy_command`), what the shield
would let through (`command`), the measured floor below the base (`floor`, about -0.6 to -0.7 m when the robot
stands) and which cameras deliver. Its commands should point along the route and slow down near people, in doors and
near shelves.

### 5. First drive

Clear space around the robot, the remote in hand, the robot standing in the mode in which the remote's sticks drive it
on its wheels.

```bash
ros2 launch deploy/b2w/launch/b2w_navigation.launch.py mode:=drive max_speed:=0.5
```

Give a goal a few meters ahead in open space, from RViz or without it:

```bash
ros2 topic pub --once /b2w/goal_relative geometry_msgs/msg/PointStamped "{point: {x: 3.0, y: 0.0}}"
```

Hold L1: the robot drives along the route; release L1: it stops. Moving a stick takes over (the policy stops sending
until the stick is released). Then longer goals, through doors, with people walking by; `max_speed` up to 1.0 m/s (the
policy was trained with caps of 0.4-1.5 m/s and is told the cap). Keep about 1.2 m from the robot while it drives:
nearer, it crawls. Where nobody walks, `crawl:=false` lets it drive at its pace near racks and walls.
`record:=$HOME/drive.npz` records every scan for `scripts/b2w/replay.py` (offline: the same commands, map statistics).

## Safety

* **Dead-man button.** The policy's commands go to the robot only while L1 is held (`deadman:=<button>` chooses
  another; check that it does nothing on its own in the robot's mode). Releasing it sends StopMove.
* **Operator takeover.** Moving a stick stops the policy's commands at once.
* **Shield** (`g1_patrol.shield`, below): the robot never drives into what its lidar, cameras and map show, keeps
  more room from tall things the faster it goes, closes in on them at most at 0.25 m/s within 0.8 m, and crawls at
  0.1 m/s within 1.2 m of anything that moves or moved there in the last two seconds (people). Crawling also happens
  near static things coming into view; `crawl:=false` drives faster where nobody walks.
* **Watchdog.** No lidar scan or odometry for 0.3 s, or no remote for 0.5 s: StopMove.
* **Software stop.** `ros2 topic pub --once /b2w_nav/stop std_msgs/msg/Bool "{data: true}"` stops the robot until
  `false` is published.
* The remote's own stop (damping) is the last resort: keep the remote in hand.

## Trying it without the robot

`deploy/b2w/sim/fake_b2w.py` is a simulated B2-W on ROS 2: it speaks like the robot (`/sportmodestate`,
`/api/sport/request`, `/wirelesscontroller`), ray-casts the Helios's point cloud in a section of the lab hall
(`sim/lab_world.json`) with walking people, and moves with the controller model of the training. With ROS 2 and
Unitree's messages sourced (e.g. on a laptop after `install.sh`), from the repository root:

```bash
bash deploy/b2w/sim/run_sil.sh 5.5 -0.6            # goal (x, y); prints "SIL RESULT {...}"
```

This runs the real launch file (odometry, SLAM, planner, policy, shield, dead-man) against the simulated robot. Results
of the release (lab section, map unknown at the start, speed cap 0.6 m/s):

| Scenario | Result | Time | Contacts (robot moving) |
|---|---|---|---|
| 12 m along the aisle, around a pallet, chairs and a tripod | reached | 42.5 s | 0 (0) |
| through a 1.0 m door into unmapped space | reached | 52.5 s | 0 (0) |
| starting facing away from the goal | reached | 35.5 s | 0 (0) |
| a person walking the aisle at the robot (ignoring it) | reached | 44.0 s | 0 (0) |
| two people crossing back and forth (ignoring it) | reached | 97.5 s | 5 (0) |
| dead-man button not held | did not move | – | 0 |

The contacts are simulated people, who ignore the robot, walking into it while it stood or crept; the robot never
moved into anyone.

## Troubleshooting

| Symptom (in the launch output or `/b2w_nav/state`) | Cause |
|---|---|
| `waiting: no odometry` and the odometry node's "no sport-mode state" | CycloneDDS on the wrong interface (`~/b2w_env.sh`), robot off, or Unitree's messages not sourced |
| `waiting: no lidar scans` | lidar driver not receiving packets (step 3) or another `lidar_topic` |
| `waiting: no remote messages` | the remote is off or not paired; without it the robot does not drive |
| `/b2w/planner_state`: `waiting for the robot's pose` | SLAM has no scans yet (lidar) or no odometry TF |
| `no route to the goal` | the goal is inside an obstacle or walled off on the map; pick another |
| state `shield` often in an empty room | the shield sees something close (`min_gap`); flickering returns near shelves |
| state `crawling (people near)` with nobody near | static things coming into view count as moving; `crawl:=false` where nobody walks |
| `floor` far from -0.6 ... -0.7 | the lidar mount in `deploy.json` is not the robot's |
| the step takes longer than 50 ms (benchmark) | `device:=cuda` with NVIDIA's PyTorch for the Jetson |

## Reference

### Interface

| Stage | Rate | Input | Output |
|---|---|---|---|
| SLAM (`slam_toolbox`) | per scan | a slice of the lidar (0.15-1.0 m above the floor), odometry | map, map -> odom |
| Route planner (`b2w.planner_node`) | per goal, then 1 Hz | map, robot pose, goal | route (`nav_msgs/Path`) |
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
of the lidar and the depth cameras per cell relative to the floor under the robot as measured in that scan (`g1_patrol.floor`), clipped to
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

### Safety shield

The shield runs on the NX after the policy and is independent of it. A command is admissible if the robot, executing it, can still stop before anything it
sees comes within a margin: it keeps the commanded arc for 0.35 s (sensing, policy, controller latency), then brakes at
0.8 m/s² (allowing 12 % more speed than commanded and 8 cm of drift per meter); the footprint is swept along this
stopping path and the command is scaled down until it comes to rest in time. Margins: 0.10 m to obstacles, 0.25 m to
drops; to tall things (people, but also machines and walls) from 0.10 m below 0.25 m/s to 0.45 m above 0.75 m/s, so
doors are passed slowly and people with room. Returns where the current scan sees something tall and the previous one
saw floor, in a cell that held nothing tall for the last second, are taken to be people walking at the robot at
1.4 m/s while it closes in on them (the second excludes the edges of furniture, which flicker between floor and tall
from scan to scan as the lidar's rays fall differently). The closing speed towards a tall thing is held to 0.25 m/s
within 0.8 m. Moving away is never limited; if the robot's own momentum already breaches, the command is zero. A person
detector on the cameras' color images can mark people (`person`), who then keep 0.45 m at any speed.

Within 1.2 m of anything tall that moves, the footprint's fastest point (turning swings the corners) crawls at 0.1 m/s
in any direction, also within 1.2 m of where something was seen moving in the last two seconds (a person close by
leaves the scan's view of what moves: the floor around them is hidden). This costs time: the moving returns also come
from static things coming into view as the robot moves, and in simulation the robot crawled in about half of all
steps, often with nobody near (results below). `crawl:=false` turns it off.

The shield scales the policy's command down, keeping its direction. When less than 30 % of it gets through and the
robot is (nearly) at rest, it also tries commands near it -- the translation or the turn alone, a nudge sideways, a
step back -- and sends the safe one that goes furthest the way the policy wanted: half through a door at an angle,
turning and driving on together may be blocked where the turn alone is not. Only at rest, because the shield checks a
command as if the robot followed it at once. A robot blocked for 1.5 s backs off for a second at 0.15 m/s, and the
policy tries again. The same code (`ShieldGuard`) runs on the robot and in the simulated evaluation.

The shield's deceleration and latency must be at least as pessimistic as the identified controller (`sysid.py fit`;
`scripts/b2w/evaluate.py --shield --shield_set latency=...,brake=...` evaluates other values). It does not see low
objects beside the robot that neither the lidar nor the cameras cover, except as remembered by the map. Tests
(`tests/test_shield.py`) cover doors, walls, drops, a person walking at the robot and closed loops through the slowest
controller of the training ranges. It does not replace the remote's stop: keep it in hand.

### Results in simulation

Evaluations of the release (`scripts/reproduce/b2w_3_evaluate.sh`; `scripts/b2w/results_table.py` prints the tables)
on the lab mixture: half the tiles are sections of the lab hall, half street and indoor tiles, robots spread over all
ten difficulty levels with up to 20 pedestrians, 1,024 robots for 300 s. Speed caps are drawn per episode from
0.4–1.5 m/s unless fixed; with the shield, the speed cap is applied before the shield, as on the robot. A pedestrian
contact counts as the robot's if it was moving towards the person at more than 0.1 m/s.

<!-- table:b2w -->
| Policy | Episodes | Goal (%) | Goal, lab hall (%) | Structure contact (%) | Pedestrian contact, robot moving (%) | Pedestrian walked into stopped robot (%) | Time-out (%) | Speed (m/s) | Shield stops (% of steps) |
|---|---|---|---|---|---|---|---|---|---|
| Lab teacher (privileged) | 18,239 | 69.4 | 79.4 | 16.6 | 8.5 | 3.2 | 1.8 | 0.53 | – |
| Fused student (lidar + 2 depth cameras) | 18,831 | 72.5 | 81.0 | 10.3 | 6.9 | 8.2 | 1.8 | 0.52 | – |
| Lidar-only student | 18,591 | 73.1 | 82.4 | 10.5 | 7.0 | 7.4 | 1.8 | 0.52 | – |
| Fused, shield | 9,339 | 57.3 | 72.6 | 1.9 | 0.6 | 22.6 | 17.6 | 0.24 | 12.4 |
| Fused, shield, cap 0.5 m/s | 8,610 | 53.9 | 69.8 | 1.8 | 0.7 | 23.9 | 19.7 | 0.21 | 11.3 |
| Fused, shield, cap 1.0 m/s | 9,609 | 58.7 | 73.6 | 1.8 | 0.4 | 22.2 | 16.8 | 0.24 | 12.9 |
<!-- /table:b2w -->

<!-- table:b2w_stress -->
| Fused student | Episodes | Goal (%) | Goal, lab hall (%) | Structure contact (%) | Pedestrian contact, robot moving (%) | Pedestrian walked into stopped robot (%) | Time-out (%) | Speed (m/s) | Shield stops (% of steps) |
|---|---|---|---|---|---|---|---|---|---|
| Nominal controller | 18,831 | 72.5 | 81.0 | 10.3 | 6.9 | 8.2 | 1.8 | 0.52 | – |
| Laggy controller (0.16-0.25 s latency) | 18,569 | 72.8 | 81.5 | 9.8 | 7.3 | 7.9 | 1.9 | 0.53 | – |
| Sluggish controller (2x response time, 1/2 acceleration) | 18,701 | 71.5 | 81.0 | 10.1 | 8.3 | 8.1 | 1.7 | 0.51 | – |
| Weak controller (70-80 % of the command) | 15,579 | 71.3 | 81.7 | 9.1 | 6.2 | 10.4 | 2.7 | 0.42 | – |
| Drifting controller (2x yaw drift) | 18,686 | 72.9 | 81.5 | 10.1 | 6.9 | 8.1 | 1.8 | 0.52 | – |
| Depth cameras failed | 19,208 | 71.3 | 80.6 | 10.6 | 7.5 | 8.9 | 1.6 | 0.53 | – |
| Degraded lidar map (20 % dropout, odometry drift) | 18,758 | 72.8 | 81.8 | 9.7 | 7.4 | 8.0 | 1.8 | 0.52 | – |
| Laggy controller, shield | 9,302 | 57.1 | 72.4 | 1.7 | 0.8 | 22.7 | 17.7 | 0.23 | 13.1 |
<!-- /table:b2w_stress -->

What the guard around the shield adds, and what crawling near anything that moved costs:

<!-- table:b2w_shield -->
| Fused student | Episodes | Goal (%) | Goal, lab hall (%) | Structure contact (%) | Pedestrian contact, robot moving (%) | Pedestrian walked into stopped robot (%) | Time-out (%) | Speed (m/s) | Shield stops (% of steps) |
|---|---|---|---|---|---|---|---|---|---|
| No shield | 18,831 | 72.5 | 81.0 | 10.3 | 6.9 | 8.2 | 1.8 | 0.52 | – |
| Shield alone: commands scaled down | 13,017 | 73.5 | 84.3 | 3.2 | 0.7 | 14.8 | 7.7 | 0.37 | 11.0 |
| + alternatives from rest, backing off (crawl:=false) | 13,056 | 76.0 | 86.7 | 2.2 | 0.8 | 14.3 | 6.7 | 0.38 | 14.1 |
| + crawling near what moved (released default) | 9,339 | 57.3 | 72.6 | 1.9 | 0.6 | 22.6 | 17.6 | 0.24 | 12.4 |
<!-- /table:b2w_shield -->

Most of the remaining pedestrian contacts with the shield are simulated pedestrians walking into a robot that has
stopped (on average the robot approached at 0.01 m/s, the pedestrian at 0.77 m/s); a quarter of the simulated
pedestrians (10 % at the easiest level, 40 % at the hardest) are distracted and give way only at the last moment. The
guard raises the goal rate of the shield alone (73.5%) to 76.0% and lowers structure contacts from 3.2% to 2.2%;
crawling near what moved lowers contacts in which the robot moves into a pedestrian from 0.8% to 0.6% and the goal
rate to 57.3%. With the speed caps of the first trials: 53.9% at 0.5 m/s, 58.7% at 1.0 m/s. The lidar-only student is
as good as the fused one in simulation; the cameras see the ground close around the robot that the lidar misses (low
obstacles beside it), which the simulated tiles rarely test.

### The lab

The released policy is specialized on the lab it is deployed in: half its training tiles are sections of that hall
(`src/g1_patrol/tasks/navigation/nyc/makerlab.py`, modeled on photographs: the painted main aisle, drawer-cabinet
benches, test machines, open racks, bins, carts, pallets, a workstation area with office chairs and camera tripods;
thin-legged objects are kept out of as a whole), and its students see the polished concrete floor lose far returns. A
scan of the actual hall turned into a twin (step 5) adds its exact layout to the evaluation.

### Nav2 instead of the built-in planner

The navigation node follows any `nav_msgs/Path` on `/plan`, so Nav2's planner server can replace ours: start the stack
with `slam:=false`, run Nav2's `map_server`, a localization and `planner_server` (`nav2_b2w.yaml`), and pass
`-p planner_action:=/compute_path_to_pose` to the navigation node to have it request a path for each goal and every
second. Nav2's `controller_server` and `bt_navigator` must not run: the navigation node takes their place.

### Controller identification (optional)

`scripts/b2w/sysid.py profile` writes a 2.5-minute command profile; play it in open space with the navigation node in
`mode:=sysid` (hold the dead-man button), then `sysid.py fit --log sysid_log.csv --json sysid.json` fits latency,
gain, deadband, response time, acceleration and yaw drift, and flags values outside the training ranges;
`scripts/b2w/evaluate.py --controller sysid:sysid.json --shield --shield_set latency=...,brake=...` evaluates the
policy and the shield with them in simulation.

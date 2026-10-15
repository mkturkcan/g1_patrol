# G1 Patrol: Humanoid Navigation in Dense Urban Pedestrian Environments

Mehmet Kerem Turkcan · [Project page](https://mehmetkeremturkcan.com/g1_patrol/)

This repository contains the simulation environments, training configurations and evaluation tools for goal-directed
navigation of a Unitree G1 humanoid in dense urban pedestrian environments. A procedurally generated suite of New
York City street scenarios (sidewalks, sidewalk sheds, crosswalks, construction zones, doorways and pedestrian
plazas, each at ten difficulty levels) is populated by pedestrians that follow a social-force model. Control is
hierarchical: a locomotion policy tracks velocity commands at 50 Hz, and a navigation policy issues these commands at
10 Hz. A navigation teacher with privileged state is trained with PPO and distilled into student policies that
observe only the robot's onboard sensors (Livox MID-360 lidar or Intel RealSense D435i depth camera). All
experiments are in simulation, using Isaac Lab 3.0 and Isaac Sim 6.1.

The project is part of work toward a civic robot system for the applications of the
[Center for Smart Streetscapes (CS3)](https://cs3-erc.org/), an NSF Engineering Research Center led by Columbia
University. It follows earlier work on street-scene perception:
[SyntheticStreetScenes](https://huggingface.co/datasets/mehmetkeremturkcan/SyntheticStreetScenes), a dataset of
approximately 5,200 generated street-scene videos, and
[StreetPrompt](https://huggingface.co/mehmetkeremturkcan/StreetPrompt), a street-camera video classifier with classes
specified as text.

![Policy architecture](docs/figures/architecture.png)

*Figure 1. Policy architecture. Numbers after the observation terms are input dimensions. The student is trained by
distillation (mean squared error between student and teacher commands); at deployment only the student path is
used.*

## Method

**Control hierarchy.** The navigation policy outputs a velocity command (forward velocity, lateral velocity, yaw
rate) at 10 Hz. The output is mapped into the command ranges by a tanh (forward −0.4 to 1.0 m/s, lateral ±0.15 m/s,
yaw rate ±1.0 rad/s) and low-pass filtered: at each step the command moves halfway to the requested value. A frozen
locomotion policy converts the command into joint position targets for the 37 joints of the G1 at 50 Hz, which are
tracked by PD controllers. The simulator runs at 200 Hz.

**Training.** Training proceeds in three stages; each stage is frozen before the next is trained.

1. *Locomotion policy* (`G1Patrol-Walk-City`). A blind walker. The policy observes only what the robot measures
   (IMU angular velocity and gravity direction, joint positions and velocities), its previous action, the velocity
   command and a gait clock, with sensor noise, sensor biases drawn per episode (IMU mounting ±1.7°, gyroscope
   ±0.02 rad/s, encoder offsets ±0.02 rad), 0–20 ms of observation latency and a history of 5 control steps; the
   critic additionally observes the base velocity and the terrain height (asymmetric actor-critic). Leg and ankle
   joints are explicit PD controllers at the gains used on the robot (hip 100, knee 150, ankle 40 Nm/rad) with the
   motors' torque and speed limits and 0–20 ms of actuation latency. Randomized per robot: ground friction (0.3–1.2),
   a torso payload (−1 to +5 kg) and center of mass (±3 cm), leg link masses (±10%), PD gains (±15%), torque limits
   (80–100% of rated, per joint), joint friction and armature; the robot is pushed every 4–8 s. Terrain: curbs and
   steps up to 20 cm, sills, potholes, broken pavement, cobblestones and ramps. Rewards: velocity tracking, a gait
   clock, foot clearance and posture, with penalties on mechanical power, foot impacts, stumbling against edges,
   changes of the joint targets and drift (the lateral-velocity and yaw-rate tracking errors, low-pass filtered over
   1 s and squared; the exponential tracking rewards hardly distinguish a sustained sideways drift of 0.1 m/s from the
   sway of the gait). Commands are resampled every 10 s: forward velocity −0.5 to 1.1 m/s, lateral ±0.3 m/s, yaw rate
   ±1.2 rad/s; 15% of the robots are commanded to stand. The policy is trained for 10,000 iterations with narrower
   randomization (friction from 0.4, torso mass ±25%, actuation latency up to 10 ms, no torque or sensor-bias
   randomization, no drift penalty), then for 8,000 iterations with the settings above, and for 3,000 more with the
   ground friction set to 0.1 and combined by maximum. MuJoCo-Warp uses the larger friction coefficient of the two
   bodies in contact, so with the default ground friction of 1.0 the robot's friction randomization had no effect and
   the earlier stages trained on friction of at least 1.0 (PhysX multiplies the coefficients). With the low ground
   coefficient the contact friction is the robot's randomized value in both simulators.
2. *Teacher* (`G1Patrol-Navigate-NYC`). Navigation with privileged observations: goal position, base velocities,
   gravity vector, positions and velocities of the 10 nearest pedestrians, obstacle distances in 72 sectors (separately
   for tall and low obstacles), a terrain height scan and the current command (332 dimensions). Training is
   initialized from an earlier teacher whose output layer is scaled by 0.3 and proceeds in stages, each warm-started
   from the previous one: the collision-risk reward (Table 2) is added at iteration 2,050 and the path-based
   pedestrian model (below) at 3,250; at 4,250 the roadway termination, the curb and personal-space margins and the
   lateral command range of ±0.15 m/s; at 6,250 the social-speed reward; at 8,750 the structure margin and speed
   terms, a discount factor of 0.995 (a 20 s horizon at 10 Hz, previously 0.99) and the final locomotion policy; at
   9,300 the present roadway and curb weights; at 10,600 the present curriculum thresholds; and at 11,850 the
   locomotion policy trained with the corrected friction. The teacher is trained for 12,650 iterations.
3. *Students* (`G1Patrol-Navigate-NYC-Lidar`, `G1Patrol-Navigate-NYC-Camera`). Distilled from the teacher with
   DAgger-style rollouts of the student; the loss is the mean squared error between the tanh-mapped actions of student
   and teacher. Each student observes proprioception, the goal position and one sensor at its mounting pose in the G1
   URDF. The sensor's returns are accumulated into a height map (8 m × 8 m, 0.2 m cells, robot-centred and
   heading-aligned): returns are binned into a world-aligned grid that moves with the robot, using the simulated robot
   motion with a random-walk error (5 mm and 0.002 rad per control step) as odometry, and a cell is cleared 3 s after it
   was last observed. The policy receives the accumulated map,
   the current scan and the previous scan (3 × 40 × 40). The camera student additionally receives the last three
   64 × 38 depth images; depth returns include distance-dependent noise, missing pixels and random occlusions.

![Training workflow](docs/figures/workflow.png)

*Figure 2. Training workflow.*

| | Locomotion policy | Teacher | Students |
|---|---|---|---|
| Algorithm | PPO | PPO | Distillation (MSE of tanh-mapped actions) |
| Simulator | Newton (MuJoCo-Warp) | Isaac Sim PhysX | Isaac Sim PhysX |
| Policy rate | 50 Hz | 10 Hz | 10 Hz |
| Parallel robots | 16,384 | 8,192 | 8,192 |
| Rollout length (steps per robot and iteration) | 24 | 32 | 32 |
| Iterations | 10,000 + 8,000 + 3,000 | 12,650 (9 stages) | 1,500 + 600 + 1,000 |
| Entropy coefficient | 0.008 | 0.001 | – |
| Network | MLP 512-256-128 (actor and critic) | MLP 512-256-128 | CNN 16-32-32 per image input, MLP 512-256-128 |
| Hardware, time | 4 GPUs, 1.9 h + 2.2 h | 4 GPUs, 21.6 h | 4 GPUs shared by both students, 12.5 h |

*Table 1. Training setup (NVIDIA RTX PRO 6000 GPUs).*

**Environment.** The terrain is a grid of 16 m × 10 m tiles; columns are the six scenario types and rows are ten
difficulty levels. In each episode the robot must traverse a route on its tile within 40 s. Pedestrians are
kinematic bodies that follow walking paths under a social-force model. A path leads from where people enter the
scene (45 m beyond the tile's ends, out of a shop door, up a subway stairwell) to 45 m beyond the tile's ends, so
pedestrians do not appear or disappear in view. Most are timed to reach the robot's route when the robot does; none
starts within 1.5 m of the robot, or within 5 m while walking towards it. A fraction of them (90% at level 1, 60% at level 10) perceive the robot, keep a clearance of at
least 0.1 m, slow down and step around it; the loss of pedestrian speed is penalized, which rewards yielding. The
others react only within 1 m. A unit test verifies that every route admits a 0.6 m wide path at every difficulty.

**Curriculum.** The curriculum keeps an exponential moving average of the success rate on each tile. A level is
unlocked once the next-easier level of the same tile column reaches 35% success (50% before teacher iteration
10,600: under the rules of Table 2 several mid-level tiles stayed below 50%, which locked all harder levels of their
column). When an episode ends, the robot's
next tile is drawn with a uniformly random scenario type and a level drawn from the unlocked levels with probability
proportional to s(1 − s) + 0.02, where s is the tile's success rate; this concentrates training on levels the policy
solves about half of the time. A fifth of the draws are uniform over the unlocked levels and 15% uniform over all
levels. The same curriculum is used for distillation, where it follows the student's success rate.

| Term | Definition | Weight |
|---|---|---|
| Progress | Reduction of distance to the goal per second, clipped to ±1.0 m/s | 2.0 |
| Goal reached | Within 0.5 m of the goal (terminates) | 100 |
| Pedestrian contact | Robot footprint (radius 0.25 m) intersects a pedestrian (radius 0.25 m) (terminates) | −200 |
| Obstacle contact | Robot footprint intersects a structure (terminates) | −150 |
| Fall | Torso contact with the ground (terminates) | −200 |
| Roadway | Robot enters the roadway outside a crosswalk (terminates) | −150 |
| Social speed | Planar speed above 0.3 m/s within 0.3 m of a pedestrian, rising linearly to 1.0 m/s at 1.5 m (m/s) | −6 |
| Personal space | Penetration of a 0.8 m clearance to the nearest pedestrian (m) | −6 |
| Collision risk | Predicted close passes within 2 s under constant relative velocity: (1 − d/0.8 m)(1 − t/2 s) per pedestrian, with t and d the time and center distance of closest approach | −4 |
| Obstacle margin | Penetration of a 0.3 m margin to the nearest structure (m) | −10 |
| Obstacle speed | Planar speed above 0.4 m/s within 0.1 m of a structure, rising linearly to 1.0 m/s at 0.4 m (m/s) | −2 |
| Pedestrian delay | Total speed lost by pedestrians slowing for the robot (m/s) | −1 |
| Curb margin | Penetration of a 0.3 m margin to the roadway (m) | −10 |
| Command rate | Squared change of the velocity command between control steps | −2.0 |
| Lateral command | Squared lateral velocity command | −1.5 |
| Backward command | Squared negative part of the forward velocity command | −2.0 |
| Yaw-rate command | Squared yaw-rate command | −0.15 |
| Policy output | Squared norm of the policy output before the tanh | −0.01 |

*Table 2. Reward terms of the navigation policy, multiplied by the control period (0.1 s).*

## Results

All navigation policies are evaluated with the curriculum disabled and robots distributed uniformly over all 120
tiles (2,400 robots, 60 s of simulated time, PhysX, sensor noise on); random pushes are disabled. An episode ends at
the goal, at the first contact with a pedestrian or a structure, when the robot enters the roadway outside a
crosswalk, on a fall, or after 40 s.

| Outcome | Teacher | Lidar student | Camera student |
|---|---:|---:|---:|
| Goal reached | 64.6% | 65.9% | 64.8% |
| Pedestrian contact | 11.2% | 11.1% | 11.2% |
| Obstacle contact | 18.1% | 18.4% | 19.4% |
| Roadway (left the sidewalk or crosswalk) | 3.4% | 1.3% | 1.9% |
| Fall | 0.31% | 0.10% | 0.14% |
| Time-out (40 s) | 2.3% | 3.2% | 2.6% |
| Mean time to goal (successful episodes) | 16.7 s | 16.7 s | 16.7 s |
| Time within 0.5 m of a pedestrian | 6.6% | 7.9% | 8.6% |
| Completed episodes | 8,375 | 8,165 | 8,376 |

*Table 3. Episode outcomes over all tiles.*

| Scenario | Easy (levels 1–4) | Medium (5–7) | Hard (8–10) |
|---|---:|---:|---:|
| Sidewalk | 94 / 96 / 95% | 63 / 74 / 67% | 16 / 19 / 14% |
| Sidewalk shed | 99 / 99 / 96% | 71 / 74 / 76% | 20 / 22 / 26% |
| Crosswalk | 91 / 93 / 93% | 55 / 55 / 60% | 49 / 57 / 53% |
| Construction zone | 80 / 74 / 75% | 27 / 20 / 33% | 1 / 0 / 2% |
| Doorway | 88 / 91 / 88% | 72 / 74 / 76% | 47 / 49 / 46% |
| Plaza | 92 / 92 / 88% | 88 / 79 / 77% | 72 / 72 / 74% |

*Table 4. Success rate by scenario type and difficulty (teacher / lidar student / camera student).*

The students reach the teacher's success rate (65.9% and 64.8% against 64.6%) and leave the roadway less often (1.3%
and 1.9% against 3.4%), at the cost of more time-outs. Success is 88–99% on easy tiles of every type except
construction zones (74–80%) and falls to 0–26% on the hard sidewalk, sidewalk-shed and construction tiles, where
walkways of about 1.0–1.8 m carry two-way foot traffic: the robot either waits for a gap that does not come within
40 s (up to 42% time-outs on hard construction tiles) or touches the structures while giving way. The previous
release reached 74.9% (teacher) and 69% (students) with lateral commands up to ±0.4 m/s, a per-step roadway penalty
in place of the roadway termination, a locomotion policy trained in effect on friction of at least 1.0, and without
the margins and speed limits of Table 2; its pedestrian contact rates were 16–20%, against 11% here, and the robots
spent 13–16% of the time within 0.5 m of a pedestrian, against 7–9%.

**Failure analysis.** For a teacher trained under most of the rules of Table 2 (iteration 8,750, 59.7% success), the
robot's state was recorded at every episode end. Contacts with structures were mostly with a structure beside the
robot (56–99% of them, depending on the tile type), at 0.7–1.0 m/s and without a pedestrian within 0.3 m in 87–100%
of the cases, and the robot's velocity typically pointed 10–20° further towards the structure than its heading. Under a
constant straight command, the locomotion policy of that time drifted sideways by 0.08–0.10 m/s, about 0.4 m within
3 s at 0.9 m/s (Table 7), a large part of the 1.0–1.5 m walkways of hard tiles. Pedestrian contacts also happened at
walking speed (median 0.87 m/s), and 21% of them within 2 s of the episode start, from pedestrians that started 1–2 m
from the robot walking towards it. The changes that followed are the drift penalty and the friction correction of
the locomotion policy; the structure margin and speed terms, the stronger social-speed term, the 20 s discount
horizon and roadway penalties no smaller than the structure penalties for the teacher (Table 2); and the spawn rule
for pedestrians. With them, success increased from 59.7% to 64.6%, pedestrian contacts decreased from 15.3% to
11.2% and structure contacts from 20.1% to 18.1%, and the mean planar speed decreased from 0.77 to 0.66 m/s.

| Quantity | Teacher, clamped commands | Teacher | Lidar student | Camera student |
|---|---:|---:|---:|---:|
| Forward command (m/s) | 1.15 | 0.68 | 0.66 | 0.67 |
| \|Lateral command\| (m/s) | 0.57 | 0.08 | 0.06 | 0.05 |
| Time with \|lateral command\| > 0.25 m/s | 96.3% | 0.0% | 0.0% | 0.0% |
| \|Yaw-rate command\| (rad/s) | 0.51 | 0.24 | 0.18 | 0.17 |
| Time with \|yaw-rate command\| > 0.8 rad/s | 25.1% | 3.7% | 0.4% | 0.2% |
| Requested command at a range limit | 68.5% | 12.0% | 1.2% | 1.0% |
| Yaw-rate command change (rad/s²) | 3.11 | 1.01 | 0.52 | 0.49 |
| Planar speed (m/s) | 1.10 | 0.66 | 0.63 | 0.64 |
| Torso tilt | 3.7° | 3.3° | 3.1° | 3.1° |
| Knee flexion | 41.3° | 36.1° | 35.5° | 35.7° |

*Table 5. Velocity commands and posture during the evaluation (time averages). A requested command is at a range
limit when it lies within 5% of the range of a command component (average over the three components). The first
column is an earlier teacher that outputs clamped commands (forward −0.6 to 1.2 m/s, lateral ±0.6 m/s, yaw rate
±1.2 rad/s) without filtering, evaluated with the same locomotion policy.*

The earlier teacher commands lateral velocities above 0.25 m/s 96% of the time and requests a range limit 69% of the
time, so the robot moves diagonally at high speed; it also leaves the roadway in 37% of its episodes, as it was
trained before the roadway termination. With the tanh mapping, the low-pass filter and the command penalties of
Table 2 the final teacher requests a range limit 12% of the time and changes the yaw-rate command three times more
slowly; the students' commands change about half as fast again.

| Metric | Walker, Newton | Walker, navigation commands | Walker, PhysX | First stage, Newton | First stage, PhysX |
|---|---:|---:|---:|---:|---:|
| Forward velocity error (m/s) | 0.094 | 0.091 | 0.100 | 0.115 | 0.118 |
| Lateral velocity error (m/s) | 0.075 | 0.066 | 0.081 | 0.093 | 0.109 |
| Yaw-rate error (rad/s) | 0.145 | 0.140 | 0.167 | 0.185 | 0.206 |
| Torso tilt | 2.8° | 2.7° | 2.8° | 4.1° | 4.1° |
| Knee flexion | 33.1° | 32.2° | 33.5° | 39.2° | 40.3° |
| Pelvis height (m) | 0.716 | 0.717 | 0.716 | 0.705 | 0.703 |
| Leg mechanical power (W) | 93 | 83 | 101 | 95 | 111 |
| Change of joint targets per step (rad) | 0.055 | 0.053 | 0.056 | 0.046 | 0.048 |
| Torso roll and pitch rate (rad/s) | 0.34 | 0.32 | 0.35 | 0.34 | 0.35 |
| Falls in 341 robot-minutes | 10 | 18 | 10 | 43 | 44 |

*Table 6. Locomotion policy on city terrain with the training randomization (payload, torque limits, latencies,
friction, sensor noise and biases): 1,024 robots for 20 s each on 200 random terrain tiles, random velocity commands
from the training ranges or, in the second column, from the navigation policy's ranges. "First stage" is the policy
after the first 10,000 iterations. Tracking errors are mean absolute errors; the other quantities are time averages.
The walker is trained on Newton; PhysX is a second simulator it was not trained on.*

| Straight command | Lateral drift (m/s) | Lateral offset after 3 s (m) | Heading change after 3 s (rad) |
|---|---:|---:|---:|
| 0.3 m/s | 0.028 (0.083) | 0.06 (0.27) | 0.08 (0.15) |
| 0.6 m/s | 0.042 (0.089) | 0.12 (0.32) | 0.11 (0.17) |
| 0.9 m/s | 0.043 (0.099) | 0.17 (0.40) | 0.15 (0.17) |

*Table 7. Drift under a constant straight command (PhysX, 256 robots per speed, after 1 s of settling); first stage in
parentheses. Lateral drift is the mean absolute 1 s average of the heading-frame lateral velocity; the offset is
measured from the line the robot started on.*

| Condition (every robot) | Walker, Newton | Walker, PhysX | First stage, Newton | First stage, PhysX |
|---|---:|---:|---:|---:|
| Training randomization | 13 | 10 | 38 | 44 |
| Torso payload +5 kg | 20 | 17 | 52 | 58 |
| Torque limits 80% of rated | 23 | 18 | 45 | 51 |
| Actuation latency 15–25 ms | 15 | 19 | 48 | 34 |
| Friction 0.3 (static) / 0.25 (dynamic) | 110 | 85 | 332 | 367 |
| Twice the sensor noise and biases | 27 | 19 | 242 | 238 |
| All of the above | 210 | 213 | 2,090 | 2,146 |

*Table 8. Falls in 341 robot-minutes (setup of Table 6) with every robot at, or beyond, the edge of the training
randomization. Counts from separate runs of the same condition differ by about their square root (Table 6 and the
first row of Table 8 are separate runs). On friction 0.3 the walker still falls about once every 3–4 robot-minutes
under these random commands.*

| Condition | Teacher | Lidar student | Camera student |
|---|---:|---:|---:|
| Nominal (Table 3) | 64.6 / 11.2 / 18.1 / 0.3 | 65.9 / 11.1 / 18.4 / 0.1 | 64.8 / 11.2 / 19.4 / 0.1 |
| Sensors: twice the noise and odometry drift, 0–200 ms height-map latency | – | 65.0 / 11.4 / 19.0 / 0.1 | 63.1 / 13.1 / 17.4 / 0.1 |
| Pedestrians: 20% faster, 30% fewer notice the robot | 56.5 / 21.0 / 17.2 / 0.3 | 56.6 / 23.3 / 16.3 / 0.1 | 54.7 / 24.4 / 17.6 / 0.1 |
| Robot: +5 kg, torque 80%, 15–25 ms actuation latency, friction 0.3 | 55.9 / 11.2 / 20.7 / 4.8 | 57.8 / 12.4 / 21.1 / 3.8 | 57.6 / 12.2 / 21.2 / 3.7 |

*Table 9. Navigation under degraded conditions applied to every robot: goal reached / pedestrian contact / obstacle
contact / fall (%). The teacher observes privileged state, so the sensor condition does not apply to it.*

## Installation

Requirements: Linux, an NVIDIA RTX GPU, Python 3.12, Isaac Sim 6.1 and Isaac Lab 3.0 (`release/3.0.0`).

```bash
conda create -n isaaclab3 python=3.12 && conda activate isaaclab3
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab-3.0
cd IsaacLab-3.0 && ./isaaclab.sh -i "newton,rl[rsl-rl],visualizer" && cd ..
pip install "moviepy>=1.0.3,<2" pytest
git clone https://github.com/mkturkcan/g1_patrol.git && pip install --no-deps -e g1_patrol
```

Set `OMNI_KIT_ACCEPT_EULA=YES` to accept the
[NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
non-interactively.

## Usage

```bash
# Stage 1: locomotion policy (Newton). The trained policy used by the navigation tasks is
# src/g1_patrol/assets/data/policies/g1_walk_city.pt.
isaaclab train --task G1Patrol-Walk-City --device cuda:0
python scripts/evaluate.py --task G1Patrol-Walk-City --checkpoint latest --full_terrain
python scripts/stress_walker.py --condition combined --checkpoint <model.pt> --num_envs 1024 --full_terrain

# Stage 2: teacher (4 GPUs, PhysX). The released teacher is initialized from an earlier teacher
# (trained with clamped commands) whose output layer is scaled by 0.3.
python scripts/warm_start.py --checkpoint <earlier teacher model.pt> --out warm.pt --scale 0.3 --std 0.5
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC --external_callback g1_patrol.register_tasks \
    --num_envs 2048 --checkpoint warm.pt physics=isaacsim_physx
python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC --checkpoint <model.pt> --num_envs 2400 --num_steps 600
python scripts/stress_nav.py --condition robot --task G1Patrol-Navigate-NYC --checkpoint <model.pt> --num_envs 2400 \
    --num_steps 600 physics=isaacsim_physx

# Stage 3: students, distilled from a teacher checkpoint
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC-Lidar --external_callback g1_patrol.register_tasks \
    --num_envs 2048 --checkpoint <teacher model.pt> physics=isaacsim_physx
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC-Camera --external_callback g1_patrol.register_tasks \
    --num_envs 2048 --checkpoint <teacher model.pt> physics=isaacsim_physx

# tests: task registration, tile capacities, route passability
python -m pytest tests
```

`evaluate_nav.py` freezes the curriculum, distributes robots uniformly over all tiles and reports outcome rates per
scenario type and difficulty. Runs are headless; select GPUs with `--device cuda:N`. Configuration values can be
overridden with Hydra arguments, e.g. `env.scene.num_envs=1024 agent.max_iterations=3000`.

**Rendering.** `scripts/demo/` renders policies with RTX: textured streets and buildings, and Isaac Sim's animated
character models in place of the pedestrian capsules. Copies of the filmed tile continue the street beyond its ends,
where pedestrians enter and leave. Only the rendering changes; pedestrian motion and policy inputs are those of the
evaluation environment. The character animations are retargeted from Isaac Sim assets and are not included in the
repository:

```bash
python scripts/demo/bake_people_animations.py --out src/g1_patrol/assets/data/people
python scripts/demo/nyc_demo.py --task G1Patrol-Navigate-NYC --checkpoint <model.pt> --no_overlay \
    --scenarios sidewalk scaffold crosswalk construction bottleneck plaza --levels 6 6 7 7 9 9 --camera chase
```

`scripts/figures/` contains the scripts for the diagrams (`make_diagrams.py`) and the overview video
(`make_video.py`).

## Tasks

| Task ID | Description |
|---|---|
| `G1Patrol-Walk-City` | Locomotion policy used by all navigation tasks (Stage 1) |
| `G1Patrol-Navigate-NYC` | Privileged navigation teacher (Stage 2) |
| `G1Patrol-Navigate-NYC-Lidar` | MID-360 lidar student (Stage 3) |
| `G1Patrol-Navigate-NYC-Camera` | D435i depth-camera student (Stage 3) |
| `G1Patrol-Walk-Flat` | Flat-ground velocity tracking |
| `G1Patrol-Walk-Street` | Earlier street-terrain walker without actuator limits and gait rewards |
| `G1Patrol-Navigate-Crowd` | Earlier flat-arena crowd navigation task with a single-plane lidar |

## Repository layout

```
src/g1_patrol/
├── assets/data/policies/        frozen locomotion policies (g1_walk_city.pt)
└── tasks/
    ├── locomotion/              walking tasks: terrains, gait and posture rewards, configurations
    └── navigation/
        ├── nyc/                 street tiles (tiles.py), terrain generator and importer
        ├── mdp/                 scenario command, observations, rewards, terminations, sensor processing
        └── config/g1/           NYC teacher and student configurations, PPO and distillation settings
scripts/
├── evaluate.py                  locomotion evaluation (tracking, posture, falls, pushes)
├── evaluate_nav.py              navigation evaluation per scenario type and difficulty, command statistics
├── export_policy.py             TorchScript export of a trained policy (e.g. the frozen walker)
├── describe_policy_io.py        observation layout, joint order, gains and actions of a task (for deployment)
├── stress_walker.py             locomotion evaluation at or beyond the edges of the training randomization
├── stress_nav.py                navigation evaluation with degraded sensors, pedestrians or robot
├── warm_start.py                checkpoint preparation for warm-started training
├── demo/                        RTX rendering with animated pedestrians
└── figures/                     diagrams and overview video
docs/                            project page (GitHub Pages)
DEPLOYMENT.md                    policy interfaces, rates, frames and the assumptions of the simulation
tests/                           registration, tile capacity and passability tests
```

## Citation

```bibtex
@misc{turkcan2026g1patrol,
  title        = {G1 Patrol: Humanoid Navigation in Dense Urban Pedestrian Environments},
  author       = {Turkcan, Mehmet Kerem},
  year         = {2026},
  howpublished = {\url{https://github.com/mkturkcan/g1_patrol}}
}
```

## License

MIT, see [LICENSE](LICENSE). Files generated from the Isaac Lab project template keep their BSD-3-Clause notice.
Icons in the figures are from [Lucide](https://lucide.dev) (ISC). Isaac Sim, Isaac Lab and their assets are subject to
their own licenses; NVIDIA assets used for rendering are downloaded from NVIDIA's asset server and are not
redistributed here.

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
rate) at 10 Hz. The output is mapped into the command ranges by a tanh (forward −0.4 to 1.0 m/s, lateral ±0.4 m/s,
yaw rate ±1.0 rad/s) and low-pass filtered: at each step the command moves halfway to the requested value. A frozen
locomotion policy converts the command into joint position targets for the 37 joints of the G1 at 50 Hz, which are
tracked by PD controllers. The simulator runs at 200 Hz.

**Training.** Training proceeds in three stages; each stage is frozen before the next is trained.

1. *Locomotion policy* (`G1Patrol-Walk-City`). Velocity tracking on uneven city terrain (potholes, cobblestones,
   curbs, curb cuts, stairs, ramps) under the torque and velocity limits of the G1 actuators, with gait-phase and
   posture rewards and random pushes. Commands are resampled every 10 s: forward velocity −0.6 to 1.2 m/s,
   lateral ±0.6 m/s, yaw rate ±1.2 rad/s; 15% of the robots are commanded to stand. The forward/lateral tracking
   reward weight is 1.0 for the first 4,000 iterations and 1.5 for the following 3,000.
2. *Teacher* (`G1Patrol-Navigate-NYC`). Navigation with privileged observations: goal position, base velocities,
   gravity vector, positions and velocities of the 10 nearest pedestrians, obstacle distances in 72 sectors (separately
   for tall and low obstacles), a terrain height scan and the current command (332 dimensions). Training is
   initialized from an earlier teacher whose output layer is scaled by 0.3. The collision-risk reward (Table 2) is
   added after 2,050 iterations, and training continues for 1,200 iterations.
3. *Students* (`G1Patrol-Navigate-NYC-Lidar`, `G1Patrol-Navigate-NYC-Camera`). Distilled from the teacher with
   DAgger-style rollouts of the student; the loss is the mean squared error between the tanh-mapped actions of student
   and teacher. Each student observes proprioception, the goal position and one sensor at its mounting pose in the G1
   URDF. The sensor's returns are accumulated into a height map (8 m × 8 m, 0.2 m cells, robot-centred and
   heading-aligned): returns are binned into a world-aligned grid that moves with the robot, using the simulated robot
   motion as odometry, and a cell is cleared 3 s after it was last observed. The policy receives the accumulated map,
   the current scan and the previous scan (3 × 40 × 40). The camera student additionally receives the last three
   64 × 38 depth images; depth returns include distance-dependent noise, missing pixels and random occlusions.

![Training workflow](docs/figures/workflow.png)

*Figure 2. Training workflow.*

| | Locomotion policy | Teacher | Students |
|---|---|---|---|
| Algorithm | PPO | PPO | Distillation (MSE of tanh-mapped actions) |
| Simulator | Newton (MuJoCo-Warp) | Isaac Sim PhysX | Isaac Sim PhysX |
| Policy rate | 50 Hz | 10 Hz | 10 Hz |
| Parallel robots | 4,096, then 16,384 | 8,192 | 8,192 |
| Rollout length (steps per robot and iteration) | 24 | 32 | 32 |
| Iterations | 4,000 + 3,000 | 2,050 + 1,200 | 1,500 |
| Entropy coefficient | 0.008 | 0.001 | – |
| Network | MLP 512-256-128 | MLP 512-256-128 | CNN 16-32-32 per image input, MLP 512-256-128 |
| Hardware, time | 1 GPU 44 min, then 4 GPUs 31 min | 4 GPUs, 6.2 h | 4 GPUs shared by both students, 5.8 h |

*Table 1. Training setup (NVIDIA RTX PRO 6000 GPUs).*

**Environment.** The terrain is a grid of 16 m × 10 m tiles; columns are the six scenario types and rows are ten
difficulty levels. In each episode the robot must traverse a route on its tile within 40 s. Pedestrians are
kinematic bodies that follow walking lanes under a social-force model. Most are timed to reach the robot's route
when the robot does. A fraction of them (90% at level 1, 60% at level 10) perceive the robot, keep a clearance of at
least 0.1 m, slow down and step around it; the loss of pedestrian speed is penalized, which rewards yielding. The
others react only within 1 m. A unit test verifies that every route admits a 0.6 m wide path at every difficulty.

**Curriculum.** The curriculum keeps an exponential moving average of the success rate on each tile. A level is
unlocked once the next-easier level of the same tile column reaches 50% success. When an episode ends, the robot's
next tile is drawn with a uniformly random scenario type and a level drawn from the unlocked levels with probability
proportional to s(1 − s) + 0.02, where s is the tile's success rate; this concentrates training on levels the policy
solves about half of the time. A fifth of the draws are uniform over the unlocked levels and 5% uniform over all
levels. The same curriculum is used for distillation, where it follows the student's success rate.

| Term | Definition | Weight |
|---|---|---|
| Progress | Reduction of distance to the goal per second, clipped to ±1.0 m/s | 2.0 |
| Goal reached | Within 0.5 m of the goal (terminates) | 100 |
| Contact | Pedestrian or obstacle contact (terminates) | −100 |
| Fall | Torso contact with the ground (terminates) | −200 |
| Personal space | Penetration of a 0.5 m clearance to the nearest pedestrian (m) | −4 |
| Collision risk | Predicted contacts within 2 s under constant relative velocity: (1 − d/0.6 m)(1 − t/2 s) per pedestrian, with t and d the time and center distance of closest approach | −4 |
| Obstacle margin | Penetration of a 0.15 m margin to the nearest structure (m) | −4 |
| Pedestrian delay | Total speed lost by pedestrians slowing for the robot (m/s) | −1 |
| Roadway | Robot in the roadway outside a crosswalk | −2 |
| Command rate | Squared change of the velocity command between control steps | −2.0 |
| Lateral command | Squared lateral velocity command | −1.5 |
| Backward command | Squared negative part of the forward velocity command | −2.0 |
| Yaw-rate command | Squared yaw-rate command | −0.15 |
| Policy output | Squared norm of the policy output before the tanh | −0.01 |

*Table 2. Reward terms of the navigation policy, multiplied by the control period (0.1 s). Contact is declared when
the robot footprint (radius 0.25 m) intersects a pedestrian (radius 0.25 m) or a structure.*

## Results

All navigation policies are evaluated with the curriculum disabled and robots distributed uniformly over all 120
tiles (2,400 robots, 60 s of simulated time, PhysX); random pushes are disabled. The earlier teacher is evaluated on
the same tiles with its own command interface: outputs clamped to forward −0.6 to 1.2 m/s, lateral ±0.6 m/s and yaw
rate ±1.2 rad/s, without filtering.

| Outcome | Teacher, clamped commands | Teacher | Lidar student | Camera student |
|---|---:|---:|---:|---:|
| Goal reached | 77.2% | 76.6% | 72.1% | 69.8% |
| Pedestrian contact | 13.1% | 15.3% | 16.1% | 17.8% |
| Obstacle contact | 8.5% | 7.5% | 11.3% | 12.3% |
| Fall | 0.9% | 0.2% | 0.1% | 0.1% |
| Time-out (40 s) | 0.3% | 0.4% | 0.4% | 0.1% |
| Mean time to goal (successful episodes) | 10.5 s | 13.3 s | 13.3 s | 13.4 s |
| Time within 0.5 m of a pedestrian | 9.9% | 11.9% | 13.9% | 14.3% |
| Completed episodes | 13,392 | 10,228 | 10,438 | 10,786 |

*Table 3. Episode outcomes over all tiles.*

| Scenario | Easy (levels 1–4) | Medium (5–7) | Hard (8–10) |
|---|---:|---:|---:|
| Sidewalk | 91 / 94 / 94% | 79 / 78 / 81% | 28 / 27 / 32% |
| Sidewalk shed | 99 / 97 / 89% | 87 / 80 / 75% | 50 / 52 / 49% |
| Crosswalk | 92 / 90 / 90% | 66 / 62 / 61% | 48 / 46 / 45% |
| Construction zone | 88 / 87 / 93% | 79 / 62 / 59% | 65 / 48 / 23% |
| Doorway | 92 / 92 / 92% | 85 / 87 / 82% | 81 / 74 / 71% |
| Plaza | 92 / 78 / 84% | 84 / 85 / 83% | 76 / 67 / 70% |

*Table 4. Success rate by scenario type and difficulty (teacher / lidar student / camera student).*

The lidar and camera students reach 94% and 91% of the teacher's success rate. Most of their additional failures are
obstacle contacts on hard construction-zone and sidewalk tiles. Two changes between the first and the final
students, accumulating the height map over 3 s and computing the distillation loss on tanh-mapped actions, reduced
obstacle contacts of the camera student from 24.4% to 12.3% and increased its success rate from 60.4% to 69.8%; the
success rate of the lidar student increased from 70.0% to 72.1%. Pedestrian contact is the most frequent failure for
all policies. On hard sidewalk tiles, where the walkway admits one person at a time, success is 27–32%.

| Quantity | Teacher, clamped commands | Teacher | Lidar student | Camera student |
|---|---:|---:|---:|---:|
| Forward command (m/s) | 1.15 | 0.94 | 0.93 | 0.94 |
| \|Lateral command\| (m/s) | 0.58 | 0.15 | 0.12 | 0.10 |
| Time with \|lateral command\| > 0.25 m/s | 96.7% | 20.6% | 8.6% | 5.9% |
| \|Yaw-rate command\| (rad/s) | 0.54 | 0.27 | 0.21 | 0.20 |
| Time with \|yaw-rate command\| > 0.8 rad/s | 28.9% | 5.4% | 1.0% | 0.7% |
| Requested command at a range limit | 70.1% | 37.4% | 28.9% | 28.5% |
| Yaw-rate command change (rad/s²) | 2.77 | 0.81 | 0.42 | 0.40 |
| Planar speed (m/s) | 1.16 | 0.88 | 0.86 | 0.87 |
| Torso tilt | 3.9° | 3.4° | 3.4° | 3.4° |
| Knee flexion | 40.2° | 38.5° | 38.4° | 38.5° |

*Table 5. Velocity commands and posture during the evaluation (time averages). A requested command is at a range
limit when it lies within 5% of the range of a command component (average over the three components).*

The earlier teacher commands lateral velocities above 0.25 m/s 97% of the time and requests a range limit 70% of the
time, so the robot moves diagonally at high speed. With the tanh mapping, the low-pass filter and the command
penalties of Table 2, these shares are 21% and 37% for the final teacher, the mean rate of change of the yaw-rate
command is 3.4 times lower, and the fall rate decreases from 0.9% to 0.2%. Success is 0.6 percentage points lower,
and successful episodes take 13.3 s instead of 10.5 s at a mean speed of 0.88 m/s instead of 1.16 m/s.

| Metric | Newton | Newton, pushes ≤ 1 m/s | PhysX |
|---|---|---|---|
| Forward velocity error (m/s) | 0.100 | 0.119 | 0.103 |
| Lateral velocity error (m/s) | 0.090 | 0.100 | 0.090 |
| Yaw-rate error (rad/s) | 0.165 | 0.179 | 0.236 |
| Torso tilt | 3.8° | 3.9° | 4.0° |
| Knee flexion | 30.6° | 31.3° | 31.5° |
| Pelvis height (m) | 0.734 | 0.736 | 0.732 |
| Falls in 85 robot-minutes | 4 | 6 | 7 |

*Table 6. Locomotion policy on city terrain (256 robots × 20 s, random velocity commands).*


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
python scripts/evaluate.py --task G1Patrol-Walk-City --checkpoint latest --push_velocity 1.0

# Stage 2: teacher (4 GPUs, PhysX). The released teacher is initialized from an earlier teacher
# (trained with clamped commands) whose output layer is scaled by 0.3.
python scripts/warm_start.py --checkpoint <earlier teacher model.pt> --out warm.pt --scale 0.3 --std 0.5
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC --external_callback g1_patrol.register_tasks \
    --num_envs 2048 --checkpoint warm.pt physics=isaacsim_physx
python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC --checkpoint <model.pt> --num_envs 2400 --num_steps 600

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
character models in place of the pedestrian capsules. Only the rendering changes; pedestrian motion and policy inputs
are those of the evaluation environment. The character animations are retargeted from Isaac Sim assets and are not
included in the repository:

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
├── warm_start.py                checkpoint preparation for warm-started training
├── demo/                        RTX rendering with animated pedestrians
└── figures/                     diagrams and overview video
docs/                            project page (GitHub Pages)
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

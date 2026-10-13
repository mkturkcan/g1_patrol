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
rate) at 10 Hz. A frozen locomotion policy converts the command into joint position targets for the 37 joints of the
G1 at 50 Hz, which are tracked by PD controllers. The simulator runs at 200 Hz.

**Training.** Training proceeds in three stages; each stage is frozen before the next is trained.

1. *Locomotion policy* (`G1Patrol-Walk-City`). Velocity tracking on uneven city terrain (potholes, cobblestones,
   curbs, curb cuts, stairs, ramps) under the torque and velocity limits of the G1 actuators, with gait-phase and
   posture rewards and random pushes. Commands: forward velocity −0.6 to 1.2 m/s, lateral ±0.6 m/s, yaw rate
   ±1.2 rad/s.
2. *Teacher* (`G1Patrol-Navigate-NYC`). Navigation with privileged observations: goal position, base velocities,
   gravity vector, positions and velocities of the 10 nearest pedestrians, obstacle distances in 72 sectors (separately
   for tall and low obstacles), a terrain height scan and the previous action (332 dimensions). A curriculum moves each
   robot to a harder tile after it reaches the goal and to an easier tile after a contact or fall.
3. *Students* (`G1Patrol-Navigate-NYC-Lidar`, `G1Patrol-Navigate-NYC-Camera`). Distilled from the teacher. Each
   student observes proprioception, the goal position and one sensor at its mounting pose in the G1 URDF: a 40 × 40
   elevation map (8 m × 8 m, 0.2 m cells) built from the MID-360 returns, or 64 × 38 depth images from the D435i with
   distance-dependent noise, missing pixels and random occlusions.

![Training workflow](docs/figures/workflow.png)

*Figure 2. Training workflow.*

| | Locomotion policy | Teacher | Students |
|---|---|---|---|
| Algorithm | PPO | PPO | Distillation (action MSE) |
| Simulator | Newton (MuJoCo-Warp) | Isaac Sim PhysX | Isaac Sim PhysX |
| Policy rate | 50 Hz | 10 Hz | 10 Hz |
| Parallel robots | 4,096, then 16,384 | 8,192 | 8,192 |
| Rollout length (steps per robot and iteration) | 24 | 32 | 32 |
| Iterations | 4,000 + 3,000 | 3,350 | 1,500 |
| Network | MLP 512-256-128 | MLP 512-256-128 | CNN 16-32-32, MLP 512-256-128 |
| Hardware, time | 1 GPU 44 min, then 4 GPUs 31 min | 4 GPUs, 4.9 h | 4 GPUs |

*Table 1. Training setup (NVIDIA RTX PRO 6000 GPUs).*

**Environment.** The terrain is a grid of 16 m × 10 m tiles; columns are the six scenario types and rows are ten
difficulty levels. In each episode the robot must traverse a route on its tile within 40 s. Pedestrians are
kinematic bodies that follow walking lanes under a social-force model. Most are timed to reach the robot's route
when the robot does. A fraction of them (90% at level 1, 60% at level 10) perceive the robot, keep a clearance of at
least 0.1 m, slow down and step around it; the loss of pedestrian speed is penalized, which rewards yielding. The
others react only within 1 m. A unit test verifies that every route admits a 0.6 m wide path at every difficulty.

| Term | Definition | Weight |
|---|---|---|
| Progress | Reduction of distance to the goal per second, clipped to ±1.5 m/s | 2.0 |
| Goal reached | Within 0.5 m of the goal (terminates) | 100 |
| Contact | Pedestrian or obstacle contact (terminates) | −100 |
| Fall | Torso contact with the ground (terminates) | −200 |
| Personal space | Penetration of a 0.5 m clearance to the nearest pedestrian (m) | −4 |
| Obstacle margin | Penetration of a 0.15 m margin to the nearest structure (m) | −4 |
| Pedestrian delay | Total speed lost by pedestrians slowing for the robot (m/s) | −1 |
| Roadway | Robot in the roadway outside a crosswalk | −2 |
| Action rate | Squared change of the velocity command | −0.1 |

*Table 2. Reward terms of the navigation policy, multiplied by the control period (0.1 s). Contact is declared when
the robot footprint (radius 0.25 m) intersects a pedestrian (radius 0.25 m) or a structure.*

## Results

The teacher is evaluated with the curriculum disabled and robots distributed uniformly over all 120 tiles (2,400
robots, 60 s, 13,238 episodes, PhysX). It reaches the goal in 79.1% of episodes; 11.9% end in pedestrian contact,
7.9% in obstacle contact, 0.7% in a fall and 0.4% in a time-out. Successful episodes take 10.5 s on average.

| Scenario | Easy (levels 1–4) | Medium (5–7) | Hard (8–10) |
|---|---|---|---|
| Sidewalk | 88.9% | 69.5% | 43.9% |
| Sidewalk shed | 97.5% | 82.4% | 50.1% |
| Crosswalk | 91.6% | 68.5% | 58.8% |
| Construction zone | 87.7% | 76.7% | 63.2% |
| Doorway | 90.4% | 89.0% | 85.2% |
| Plaza | 88.2% | 83.0% | 84.0% |

*Table 3. Teacher success rate by scenario type and difficulty.*

Pedestrian contacts are mostly initiated by the robot: in 67.8% of them the robot approached faster than the
pedestrian. Success is lowest on hard sidewalk and sidewalk-shed tiles, where the walkway admits one person at a time.

| Metric | Newton | Newton, pushes ≤ 1 m/s | PhysX |
|---|---|---|---|
| Forward velocity error (m/s) | 0.100 | 0.119 | 0.103 |
| Lateral velocity error (m/s) | 0.090 | 0.100 | 0.090 |
| Yaw-rate error (rad/s) | 0.165 | 0.179 | 0.236 |
| Torso tilt | 3.8° | 3.9° | 4.0° |
| Knee flexion | 30.6° | 31.3° | 31.5° |
| Pelvis height (m) | 0.734 | 0.736 | 0.732 |
| Falls in 85 robot-minutes | 4 | 6 | 7 |

*Table 4. Locomotion policy on city terrain (256 robots × 20 s, random velocity commands).*

The lidar student is being trained; the camera student has not been trained yet.

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

# Stage 2: teacher (4 GPUs, PhysX)
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC --external_callback g1_patrol.register_tasks \
    --num_envs 2048 physics=isaacsim_physx
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
| `G1Patrol-Walk-Flat`, `G1Patrol-Walk-Rough` | Flat- and rough-terrain velocity tracking; `Walk-Rough` is the base configuration of `Walk-City` |
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
├── evaluate_nav.py              navigation evaluation per scenario type and difficulty
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

# G1 Patrol: a civic humanoid for New York streetscapes

Reinforcement-learning tasks, built on [Isaac Lab 3.0](https://github.com/isaac-sim/IsaacLab) and Isaac Sim 6.1, that
teach a [Unitree G1](https://www.unitree.com/g1) humanoid to walk upright on city pavement and to make its way
through crowded New York sidewalks, scaffolding tunnels, crosswalks and construction detours, giving way to
pedestrians, using only the sensors the real robot carries.

## Why

The idea of this project is to build a **civic robot system** for the applications of the
[Center for Smart Streetscapes (CS3)](https://cs3-erc.org/), the NSF Engineering Research Center led by Columbia
University that works toward livable and safe streetscapes. A robot that shares the sidewalk with people (to patrol,
inspect, report hazards or assist) has to move like a considerate pedestrian first: keep its balance on broken
pavement and curbs, fit through tight spaces, and step aside for people rather than expect them to step aside for it.

It follows earlier streetscape work in the same direction:

- [**SyntheticStreetScenes**](https://huggingface.co/datasets/mehmetkeremturkcan/SyntheticStreetScenes): a dataset of
  ~5,200 AI-generated street-scene videos (traffic, crashes, floods, civic events, ...) with prompts and attributes;
- [**StreetPrompt**](https://huggingface.co/mehmetkeremturkcan/StreetPrompt): a street-camera video classifier whose
  classes you write as text, trained on SyntheticStreetScenes and captioned real traffic footage.

Those projects let a city *see* its streets; this one is about a robot that can *walk* them. Everything here is
simulation only so far.

## What is in here

| Stage | Task ID | What it learns | Status |
|---|---|---|---|
| 1 | `G1Patrol-Walk-Flat` | Walk on flat ground following (vx, vy, yaw-rate) commands | trained |
| 2 | `G1Patrol-Walk-Rough` | Walk on rough terrain with a height scan | smoke-tested |
| 3 | `G1Patrol-Walk-Street` | Curbs, sills, stairs, ramps, rough pavement; shoves of up to 1.2 m/s | trained |
| 3b | `G1Patrol-Walk-City` | **Upright, realistic walker**: real motor torque/speed limits, gait-phase rewards, potholes, cobbles, curbs, curb cuts | trained, used by all navigation tasks |
| 4 | `G1Patrol-Navigate-Crowd` | Reach a goal through walking people, poles and walls (flat arena) | trained |
| 5 | `G1Patrol-Navigate-NYC` | **Privileged teacher** for six New York scenario types at ten difficulty levels | trained |
| 6 | `G1Patrol-Navigate-NYC-Lidar` | Student distilled from the teacher, seeing the world only through the G1's **Livox MID-360** lidar | pipeline tested |
| 7 | `G1Patrol-Navigate-NYC-Camera` | Camera-only student using the G1's **RealSense D435i** depth camera | pipeline tested |

The navigation policies are hierarchical: a navigator at 10 Hz outputs velocity commands to the frozen
`G1Patrol-Walk-City` walker, which turns them into joint targets at 50 Hz and handles balance, curbs and shoves.

## New York scenarios

`G1Patrol-Navigate-NYC` trains on a procedurally generated grid of 16 m × 10 m street tiles: columns are scenario
types, rows are difficulty levels, and a curriculum moves each robot to harder tiles when it reaches its goal and to
easier ones when it collides or falls.

| Scenario | What makes it hard |
|---|---|
| Busy sidewalk | Two-way foot traffic on a sidewalk that narrows from 5 m to 2.6 m; hydrants, poles, parking meters, newsboxes, bike racks and low tree-pit fences along the curb; people stepping out of doorways |
| Sidewalk shed (scaffolding) | A tunnel of 9 cm poles and rails narrowing to 1.25 m; oncoming people force the robot to wait in a gap or doorway |
| Crossing the street | Curbs or curb cuts, potholes, waves of people crossing towards the robot; walking in the street outside the crosswalk is penalized |
| Construction detour | A chicane of fences and jersey barriers over steel plates and broken pavement, one person wide at the hardest level |
| Doorway rush | A partition with a single opening or a subway entrance with people streaming out |
| Times Square plaza | Dense criss-crossing crowds, tourists standing around, planters, tables and kiosks |

Pedestrians are kinematic bodies driven by a social-force model. Most are timed to reach the robot's route when the
robot does, so the crowd really does cross and block its path. Pedestrians who notice the robot keep their distance,
slow down and step around it; that slowdown is penalized, which is what teaches the robot to **give way**. Distracted
pedestrians (on their phone) only swerve at the last moment, so the robot has to keep out of their path. Every tile is
checked by a unit test to be passable for the robot at every difficulty.

### Sensors

The students use only Unitree's official sensor setup, mounted where the G1's URDF puts it:

- **Livox MID-360** lidar, upside down in the head (~1.13 m up, −52° to +7° vertical field of view): its returns are
  binned into a robot-centred 8 m × 8 m height map that shows curbs, potholes, low fences, poles and people;
- **RealSense D435i** depth camera, pitched 47.6° down: degraded with distance-dependent noise, missing pixels and
  random occlusions. With that tilt it sees the ground from about 0.3 m to 3.4 m ahead, so a camera-only robot sees
  people late; a re-angled bracket or an extra forward camera would help.

The teacher instead observes the exact positions and velocities of the ten nearest pedestrians, obstacle distances in
72 directions and the terrain, and is distilled into the students with rsl-rl's `DistillationRunner`.

## Results

All numbers are from headless evaluation in play mode (no random shoves unless stated).

### City walker (`G1Patrol-Walk-City`)

Compared with the earlier street walker over 256 robots × 20 s (85 robot-minutes) on city terrain, the city walker
stands and walks more upright while tracking velocity commands as well as before:

| | Street walker | City walker |
|---|---|---|
| Torso tilt | 6.4° | 3.8° |
| Knee bend | 41° | 30° |
| Pelvis height | 0.69 m | 0.73 m |
| Falls (Newton, its training engine) | — | 4 |
| Falls with shoves of up to 1 m/s | — | 6 |
| Falls on PhysX | — | 7 |

### New York teacher (`G1Patrol-Navigate-NYC`)

3350 PPO iterations (4.9 h on 4 GPUs, 4 × 2048 robots, PhysX). Evaluated with `scripts/evaluate_nav.py` with the
curriculum frozen and robots spread evenly over all tiles (2400 robots × 60 s, 13,238 episodes). Share of episodes
that reach the goal, by difficulty (easy: levels 1–4, medium: 5–7, hard: 8–10):

| Scenario | Easy | Medium | Hard |
|---|---|---|---|
| Busy sidewalk | 89 % | 69 % | 44 % |
| Sidewalk shed (scaffolding) | 97 % | 82 % | 50 % |
| Crossing the street | 92 % | 68 % | 59 % |
| Construction detour | 88 % | 77 % | 63 % |
| Doorway rush | 90 % | 89 % | 85 % |
| Times Square plaza | 88 % | 83 % | 84 % |

Overall, **79 %** of episodes reach the goal; the rest end in a pedestrian collision (12 %), an obstacle collision
(8 %), a fall (0.7 %) or a time-out (0.4 %). Successful episodes take 10.5 s on average. Two thirds of the pedestrian collisions happen while the robot is
closing in faster than the pedestrian, so the robot is still too assertive in the densest crowds; the hardest
sidewalk and scaffolding tiles, where there is no room to pass oncoming people, remain the main open problem.

### Earlier stages

- **Flat walker:** 1500 PPO iterations (11.5 min on Newton, 4096 robots); velocity errors of 0.06 m/s and no falls
  in 85 robot-minutes. Policies transfer between Newton and PhysX once the joint and body order is pinned.
- **Street walker:** 5000 iterations on Newton; 3 falls in 85 robot-minutes with or without 1 m/s shoves, on either
  engine.
- **Crowd navigator** (flat arena, 10 pedestrians, 8 poles, 4 walls): 83 % of episodes reach the goal, 15 % end in a
  pedestrian collision; in most of those the pedestrian walked into a robot that was backing away, which motivated
  the more realistic pedestrian behaviour of the New York tasks.

## Installation

Requires an NVIDIA RTX GPU, Python 3.12, Isaac Sim 6.1 and Isaac Lab 3.0 (`release/3.0.0` branch).

```bash
conda create -n isaaclab3 python=3.12 && conda activate isaaclab3
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab-3.0
cd IsaacLab-3.0 && ./isaaclab.sh -i "newton,rl[rsl-rl],visualizer" && cd ..
pip install "moviepy>=1.0.3,<2" pytest
git clone https://github.com/mkturkcan/g1_patrol.git && pip install --no-deps -e g1_patrol
```

Isaac Sim asks you to accept the [NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
on first launch (set `OMNI_KIT_ACCEPT_EULA=YES` to accept it non-interactively).

## Usage

```bash
isaaclab list_envs --show_presets                      # tasks and physics presets

# walkers (Newton is ~2x faster for these)
isaaclab train --task G1Patrol-Walk-City --device cuda:0
python scripts/evaluate.py --task G1Patrol-Walk-City --checkpoint latest --push_velocity 1.0   # shove test

# New York teacher: 4 GPUs, PhysX (kinematic pedestrians are slow on MuJoCo-Warp)
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC --external_callback g1_patrol.register_tasks \
    --num_envs 2048 physics=isaacsim_physx
python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC --checkpoint latest --num_envs 2400   # per-scenario table

# students: distill the teacher into the lidar-only, then the camera-only policy
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC-Lidar --external_callback g1_patrol.register_tasks \
    --num_envs 2048 --checkpoint <teacher model_XXXX.pt> physics=isaacsim_physx
isaaclab train_multigpu --num_gpus 4 --task G1Patrol-Navigate-NYC-Camera --external_callback g1_patrol.register_tasks \
    --num_envs 2048 --checkpoint <teacher model_XXXX.pt> physics=isaacsim_physx

python -m pytest tests                                  # registrations and tile passability
```

Notes: runs are headless unless you pass `--visualizer kit`; pick GPUs with `--device cuda:N`; `--checkpoint`
accepts `latest`, `best`, a run directory or a `model_XXXX.pt` file; config values can be overridden with Hydra, e.g.
`env.scene.num_envs=1024 agent.max_iterations=3000`.

### Demo videos

`scripts/demo/` renders the policies in dressed-up New York streets with RTX: textured asphalt, concrete sidewalks
with granite curbs, zebra crossings, brick facades with shop windows, green sidewalk sheds, street trees, a skyline,
and Isaac Sim's animated people instead of the capsules the robot trains with. Pedestrians' walk cycles follow their
simulated speed, and a chase camera films one robot per scenario. Only the rendering changes; the robot, the people's
motion and what the policy sees are exactly those of the evaluation environment.

```bash
# once: retarget Isaac Sim's walk/idle clips onto the People characters (not redistributed here)
python scripts/demo/bake_people_animations.py --out src/g1_patrol/assets/data/people

# one clip per episode, named after its scenario and outcome
python scripts/demo/nyc_demo.py --task G1Patrol-Navigate-NYC --checkpoint <model.pt> \
    --scenarios sidewalk scaffold crosswalk construction bottleneck plaza --level 7 --seconds 60
```

## Layout

```
src/g1_patrol/
├── assets/data/policies/        # frozen walkers used by the navigation tasks (g1_walk_city.pt, g1_walk_street.pt)
└── tasks/
    ├── locomotion/              # G1Patrol-Walk-*: terrains, gait/posture rewards, push curriculum, configs
    └── navigation/
        ├── nyc/                 # street tiles (tiles.py, pure numpy), terrain generator and importer
        ├── mdp/nyc_scenario.py  # routes, pedestrian traffic, privileged state
        ├── mdp/nyc_terms.py     # observations, rewards, terminations, curriculum
        ├── mdp/perception.py    # lidar height map and degraded depth images for the students
        ├── mdp/actions.py       # drives the frozen walker with clamped velocity commands
        ├── mdp/crowd.py         # the earlier flat-arena crowd task
        └── config/g1/           # nyc_env_cfg.py (teacher and students), crowd_env_cfg.py, PPO/distillation configs
scripts/
├── evaluate.py                  # walkers: tracking error, posture, falls, shove test (+ RTX video)
├── evaluate_nav.py              # navigators: outcome rates per scenario and difficulty
├── render_trajectory.py         # offline stick-figure video
└── demo/                        # RTX demo: city looks, animated people, chase camera
tests/                           # task registrations, tile capacity and passability
```

## License

MIT, see [LICENSE](LICENSE). Files generated from the Isaac Lab project template keep their original BSD-3-Clause
notice. Isaac Sim, its assets (robots, people, materials) and Isaac Lab are subject to their own licenses; the demo
downloads the assets it uses from NVIDIA's asset server rather than redistributing them.

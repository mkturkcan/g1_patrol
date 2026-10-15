# Deployment interface

This note specifies what the trained policies consume and produce, at which rates and in which frames, and which
properties of the real robot the simulation assumed. It is generated in part by `scripts/describe_policy_io.py`
(`--task G1Patrol-Walk-City`, `--task G1Patrol-Navigate-NYC-Lidar`, `--task G1Patrol-Navigate-NYC-Camera`), which
prints the exact tables for the current configuration. Nothing here has been tested on hardware.

## Control hierarchy

| Stage | Rate | Input | Output |
|---|---|---|---|
| Navigator (student) | 10 Hz | goal, state estimate, height map (lidar or depth) | 3 raw actions |
| Command shaping | 10 Hz | raw actions | velocity command `(vx, vy, wz)` |
| Walker | 50 Hz | IMU, joint encoders, command, gait clock (5 steps) | 37 joint position targets |
| Joint PD | 200 Hz in simulation (motor drivers on the robot) | targets | torques |

Command shaping: `c = lo + (tanh(a) + 1) / 2 · (hi − lo)` per component with ranges `vx ∈ [−0.4, 1.0]` m/s,
`vy ∈ [−0.15, 0.15]` m/s, `wz ∈ [−1.0, 1.0]` rad/s, then a first-order low-pass `c_t = 0.5 c_{t−1} + 0.5 c`. The walker
was trained on the wider ranges `vx ∈ [−0.5, 1.1]`, `vy ∈ [−0.3, 0.3]`, `wz ∈ [−1.2, 1.2]`, with 15% standing commands.

## Walker (`policies/g1_walk_city.pt`, TorchScript)

Input: 610 values, five frames (the current one last) of seven terms, concatenated term by term (all five frames of
`base_ang_vel`, then all five of `projected_gravity`, ...). An observation normalizer is part of the exported module.

| Term | Size | Frame / units | Noise in training (per step; bias per episode) | Latency |
|---|---|---|---|---|
| `base_ang_vel` | 3 | pelvis IMU gyro, rad/s | ±0.2; bias ±0.02 | 0–20 ms |
| `projected_gravity` | 3 | gravity unit vector in the pelvis frame | ±0.05; bias ±0.03 (≈1.7° mounting) | 0–20 ms |
| `velocity_commands` | 3 | `(vx, vy, wz)` in the heading frame | – | – |
| `joint_pos` | 37 | encoder minus default position, rad | ±0.01; offset ±0.02 | 0–20 ms |
| `joint_vel` | 37 | rad/s | ±1.5 | 0–20 ms |
| `actions` | 37 | the walker's previous output | – | – |
| `gait_phase` | 2 | `(sin, cos)(2π t / 0.8 s)`, free-running clock | – | – |

Output: 37 actions; joint position target = default position + 0.25 · action, for all joints in the order below
(legs, torso, arms, hands; the hand targets only hold the hands). The walker does not use the base linear velocity or
any exteroception; the critic used both during training only.

Joint order, default position (rad), PD gains (Nm/rad, Nm·s/rad) and torque limit (Nm):

| Joints | Default | Kp | Kd | Limit |
|---|---|---|---|---|
| 0–5 left hip pitch, roll, yaw, knee, ankle pitch, roll | −0.20, 0, 0, 0.42, −0.23, 0 | 100 (hip), 150 (knee), 40 (ankle) | 2, 4, 2 | 88, 139, 50 |
| 6–11 right leg, same order | same | same | same | same |
| 12 torso | 0 | 200 | 5 | 88 |
| 13–17 left shoulder pitch, roll, yaw, elbow pitch, roll | 0.35, 0.16, 0, 0.87, 0 | 40 | 10 | 25 |
| 18–24 left hand (five, six, three, four, zero, one, two) | 0, 0, 0, 0, 0, 1.0, 0.52 | 40 | 10 | 5 |
| 25–29 right arm | 0.35, −0.16, 0, 0.87, 0 | 40 | 10 | 25 |
| 30–36 right hand | 0, 0, 0, 0, 0, −1.0, −0.52 | 40 | 10 | 5 |

The model is Isaac Lab's 37-DoF G1 with three-finger hands (`G1_MINIMAL_CFG`). Robots with a different arm or hand
configuration (e.g. the 29-DoF G1) need the walker retrained on that model; the leg interface is the same.

## Navigator (students)

Input groups (see `describe_policy_io.py` for exact shapes):

- `student` (14): goal position in the heading frame (x, y; m), base linear velocity (body frame, m/s), base angular
  velocity (rad/s), projected gravity, previous raw actions (3). In training the goal has ±0.1 m noise; the velocities
  ±0.1 m/s / ±0.2 rad/s noise and a per-episode bias of ±0.05 m/s / ±0.02 rad/s (state-estimator error).
- `lidar_map` (3 × 40 × 40), lidar student: Livox MID-360 returns binned into a 0.2 m grid of 8 × 8 m around the
  robot in its heading frame (rows forward, columns left); channels: map accumulated over 3 s (highest return of the
  latest scan per cell, forgotten 3 s after last seen), current scan, previous scan (motion-compensated). Heights are
  relative to the ground under the robot. The robot's motion between scans comes from odometry; in training the pose
  estimate drifts as a random walk of 5 mm and 0.002 rad per step, the range noise is 0.5% of the distance and the
  map is 0–100 ms old. The MID-360 is mounted upside down on the torso (field of view −52° to 7°).
- `depth` (3 × 38 × 64) and `depth_map` (3 × 40 × 40), camera student: the RealSense D435i depth image (87° × 58°,
  downsampled to 64 × 38, clipped at 6 m) for the last three frames, and the same accumulated height map built from the
  depth returns.

Output: 3 raw actions, shaped as above.

## What the simulation assumed

Ranges sampled per robot in training (walker and navigator); the stress tests (`scripts/stress_walker.py`,
`scripts/stress_nav.py`) evaluate fixed values at or beyond their edges.

| Property | Training range |
|---|---|
| Actuation latency | 0–20 ms |
| Observation latency | 0–20 ms (walker), 0–100 ms (height maps) |
| Torque limit | 80–100% of rated, per joint |
| PD gains | ±15% |
| Joint friction / armature | 0–0.3 Nm / ×0.8–1.2 |
| Torso mass | −1 to +5 kg; center of mass ±3 cm |
| Link masses (pelvis, legs) | ±10% |
| Ground friction (static / dynamic) | 0.3–1.2 / 0.25–1.0 |
| Shoves | every 4–8 s (walker), 6–10 s (navigation) |
| Terrain | curbs up to 20 cm, sills, potholes, cobblestones, ramps; no stairs |

Not modelled: motor heating and the torque-speed curve beyond the velocity limits, foot and joint compliance, cable
and battery dynamics, the MID-360's non-repetitive scan pattern (approximated by 32 channels × 360 directions),
depth-camera artefacts beyond dropout and range noise, glass and other non-returning surfaces, traffic signals.

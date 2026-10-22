#!/bin/bash
# B2-W stage 2: lidar student (the RoboSense lidar accumulated into a height map, odometry, the goal), distilled from
# the teacher with DAgger-style rollouts of the student, 4,096 robots on one GPU (B2W_DEVICE, default cuda:0).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
ITERATIONS=${ITERATIONS:-1500}
TEACHER=$(latest "$LOGS/b2w_nav_teacher/$TAG")
isaaclab train --rl_library rsl_rl --external_callback g1_patrol.register_tasks --seed "$SEED" --run_timestamp "$TAG" \
    --task G1Patrol-B2W-Navigate-Lidar --num_envs 4096 --device "$DEVICE" --checkpoint "$TEACHER" \
    agent.max_iterations="$ITERATIONS" physics=isaacsim_physx > "$OUT/b2w_2_student.log" 2>&1
echo "B2-W lidar student: $(latest "$LOGS/b2w_nav_lidar_student/$TAG")"

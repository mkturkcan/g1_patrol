#!/bin/bash
# B2-W stage 1b: the teacher specialized on the lab it is deployed in -- fine-tuned from the general teacher on a
# mixture of lab sections (half the tiles; src/g1_patrol/tasks/navigation/nyc/makerlab.py) and the general street and indoor tiles, 1,000
# iterations, 4,096 robots on one GPU (B2W_DEVICE, default cuda:0).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
ITERATIONS=${ITERATIONS:-1000}
TEACHER=$(latest "$LOGS/b2w_nav_teacher/$TAG")
isaaclab train --rl_library rsl_rl --external_callback g1_patrol.register_tasks --seed "$SEED" --run_timestamp "$TAG" \
    --task G1Patrol-B2W-Navigate-Lab --num_envs 4096 --device "$DEVICE" --checkpoint "$TEACHER" \
    agent.max_iterations="$ITERATIONS" physics=isaacsim_physx > "$OUT/b2w_1b_lab_teacher.log" 2>&1
echo "B2-W lab teacher: $(latest "$LOGS/b2w_nav_lab_teacher/$TAG")"

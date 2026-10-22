#!/bin/bash
# B2-W stage 1: navigation teacher for the Unitree B2-W, driven through its built-in controller (a randomized model of
# its velocity tracking), on New York street tiles and indoor tiles, with privileged observations; trained from scratch
# with PPO on PhysX, 4,096 robots on one GPU (B2W_DEVICE, default cuda:0).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
ITERATIONS=${ITERATIONS:-3000}
isaaclab train --rl_library rsl_rl --external_callback g1_patrol.register_tasks --seed "$SEED" --run_timestamp "$TAG" \
    --task G1Patrol-B2W-Navigate --num_envs 4096 --device "$DEVICE" agent.max_iterations="$ITERATIONS" \
    physics=isaacsim_physx > "$OUT/b2w_1_teacher.log" 2>&1
echo "B2-W teacher: $(latest "$LOGS/b2w_nav_teacher/$TAG")"

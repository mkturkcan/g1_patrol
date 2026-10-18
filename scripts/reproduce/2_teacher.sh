#!/bin/bash
# Stage 2: navigation teacher with privileged observations, trained from scratch with PPO on PhysX, 8,192 robots on 4
# GPUs, 10,000 iterations. Executes the walker exported by stage 1.
source scripts/reproduce/env.sh
"${TRAIN[@]}" --master_port 29602 --task G1Patrol-Navigate-NYC --num_envs 2048 agent.max_iterations=10000 \
    physics=isaacsim_physx > "$OUT/2_teacher.log" 2>&1
echo "teacher: $(latest "$LOGS/g1_nyc_teacher/$TAG")"

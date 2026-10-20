#!/bin/bash
# Civic stage 3: lidar (Livox MID-360 for people and the height map, D435i color for litter) and camera-only (D435i)
# students, distilled from the civic teacher with DAgger-style rollouts of the student, 8,192 robots on 4 GPUs.
source scripts/reproduce/env.sh
ITERATIONS=${ITERATIONS:-2000}
TEACHER=$(latest "$LOGS/g1_civic_teacher/$TAG")
"${TRAIN[@]}" --master_port 29606 --task G1Patrol-Navigate-Civic-Lidar --num_envs 2048 --checkpoint "$TEACHER" \
    agent.max_iterations="$ITERATIONS" physics=isaacsim_physx > "$OUT/7_civic_lidar.log" 2>&1
"${TRAIN[@]}" --master_port 29607 --task G1Patrol-Navigate-Civic-Camera --num_envs 2048 --checkpoint "$TEACHER" \
    agent.max_iterations="$ITERATIONS" physics=isaacsim_physx > "$OUT/7_civic_camera.log" 2>&1
echo "civic lidar student: $(latest "$LOGS/g1_civic_lidar_student/$TAG")"
echo "civic camera student: $(latest "$LOGS/g1_civic_camera_student/$TAG")"

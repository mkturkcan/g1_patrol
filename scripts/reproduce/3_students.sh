#!/bin/bash
# Stage 3: lidar (Livox MID-360) and depth-camera (RealSense D435i) students, distilled from the stage-2 teacher with
# DAgger-style rollouts of the student, 8,192 robots on 4 GPUs, 1,500 iterations each.
source scripts/reproduce/env.sh
TEACHER=$(latest "$LOGS/g1_nyc_teacher/$TAG")
"${TRAIN[@]}" --master_port 29603 --task G1Patrol-Navigate-NYC-Lidar --num_envs 2048 --checkpoint "$TEACHER" \
    agent.max_iterations=1500 physics=isaacsim_physx > "$OUT/3_lidar.log" 2>&1
"${TRAIN[@]}" --master_port 29604 --task G1Patrol-Navigate-NYC-Camera --num_envs 2048 --checkpoint "$TEACHER" \
    agent.max_iterations=1500 physics=isaacsim_physx > "$OUT/3_camera.log" 2>&1
echo "lidar student: $(latest "$LOGS/g1_nyc_lidar_student/$TAG")"
echo "camera student: $(latest "$LOGS/g1_nyc_camera_student/$TAG")"

#!/bin/bash
# B2-W stage 4: the camera student, the fused student that also sees the front color camera's image (features of a
# frozen DINOv2-small), distilled from the lab teacher with the camera seeing the training environment drawn as in the
# demo (G1Patrol-B2W-Navigate-Lab-Camera); it starts from the fused student of stage 2 (the new inputs' weights at
# zero). 256 robots on one GPU (B2W_CAMERA_DEVICE, default cuda:0). Run after scripts/reproduce/b2w.sh.
source scripts/reproduce/env.sh
DEVICE=${B2W_CAMERA_DEVICE:-cuda:0}
ITERATIONS=${ITERATIONS:-300}
TEACHER=$(latest "$LOGS/b2w_nav_lab_teacher/$TAG")
STUDENT=$(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")
isaaclab train --rl_library rsl_rl --external_callback g1_patrol.register_tasks --seed "$SEED" --run_timestamp "$TAG" \
    --task G1Patrol-B2W-Navigate-Lab-Camera --num_envs 256 --device "$DEVICE" --checkpoint "$TEACHER" \
    agent.max_iterations="$ITERATIONS" agent.algorithm.student_init="$PWD/$STUDENT" physics=isaacsim_physx \
    > "$OUT/b2w_4_camera_student.log" 2>&1
echo "B2-W camera student: $(latest "$LOGS/b2w_nav_lab_camera_student/$TAG")"

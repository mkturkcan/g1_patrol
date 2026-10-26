#!/bin/bash
# B2-W stage 2: students distilled from the lab teacher with DAgger-style rollouts of the student on the lab mixture
# (polished floor: far floor returns go missing), 4,096 robots each on one GPU: the release student with the lidar and
# the front and rear RealSense D435i depth cameras fused into one height map (B2W_DEVICE, default cuda:0), and a
# fallback with the lidar only (B2W_DEVICE2, default: after the first, on the same GPU).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
DEVICE2=${B2W_DEVICE2:-}
ITERATIONS=${ITERATIONS:-1500}
TEACHER=$(latest "$LOGS/b2w_nav_lab_teacher/$TAG")
student() {  # task, device, log
    isaaclab train --rl_library rsl_rl --external_callback g1_patrol.register_tasks --seed "$SEED" --run_timestamp "$TAG" \
        --task "$1" --num_envs 4096 --device "$2" --checkpoint "$TEACHER" agent.max_iterations="$ITERATIONS" \
        physics=isaacsim_physx > "$OUT/$3" 2>&1
}
if [ -n "$DEVICE2" ]; then
    student "G1Patrol-B2W-Navigate-${B2W_LAB:-Lab}-Lidar" "$DEVICE2" b2w_2_lidar_student.log &
    LIDAR_PID=$!
fi
student "G1Patrol-B2W-Navigate-${B2W_LAB:-Lab}-Fused" "$DEVICE" b2w_2_fused_student.log
if [ -n "$DEVICE2" ]; then wait "$LIDAR_PID"; else student "G1Patrol-B2W-Navigate-${B2W_LAB:-Lab}-Lidar" "$DEVICE" b2w_2_lidar_student.log; fi
echo "B2-W fused student: $(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")"
echo "B2-W lidar student: $(latest "$LOGS/b2w_nav_lab_lidar_student/$TAG")"

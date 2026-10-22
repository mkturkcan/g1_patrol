#!/bin/bash
# B2-W stage 3: evaluations of the teacher and the lidar student (nominal controller; controller stress tests; a
# degraded lidar map) and the export of the student for the robot (deploy/b2w/release: policy.onnx, deploy.json,
# samples.npz). One GPU (B2W_DEVICE, default cuda:0).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
STEPS=${STEPS:-3000}
TEACHER=$(latest "$LOGS/b2w_nav_teacher/$TAG")
STUDENT=$(latest "$LOGS/b2w_nav_lidar_student/$TAG")
EVAL=(python scripts/b2w/evaluate.py --num_envs 1024 --num_steps "$STEPS" --device "$DEVICE" physics=isaacsim_physx)
"${EVAL[@]}" --task G1Patrol-B2W-Navigate --checkpoint "$TEACHER" > "$OUT/b2w_3_eval_teacher.log" 2>&1
for controller in nominal laggy sluggish weak drifty; do
    "${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lidar --checkpoint "$STUDENT" --controller "$controller" --map_stats \
        > "$OUT/b2w_3_eval_student_$controller.log" 2>&1
done
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lidar --checkpoint "$STUDENT" --lidar_dropout 0.2 --odometry_drift 0.01 \
    --out "$(dirname "$STUDENT")/eval/b2w_degraded_map.json" > "$OUT/b2w_3_eval_student_degraded.log" 2>&1
python scripts/b2w/export.py --checkpoint "$STUDENT" --out deploy/b2w/release --device "$DEVICE" physics=isaacsim_physx \
    > "$OUT/b2w_3_export.log" 2>&1
echo "B2-W evaluations: $(dirname "$STUDENT")/eval, exported policy: deploy/b2w/release"

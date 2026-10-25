#!/bin/bash
# B2-W stage 3: evaluations on the lab mixture -- the lab teacher; the fused student with the nominal controller, with
# controller stress tests, with the safety shield in the loop (also at the robot's first speed caps, 0.5 and 1 m/s;
# also without crawling near what moved, and the shield alone),
# with failed depth cameras and a degraded map; the lidar student -- and the export of both students for the robot
# (deploy/b2w/release: policy.onnx, deploy.json, samples.npz; deploy/b2w/release_lidar). One GPU (B2W_DEVICE, default
# cuda:0).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
STEPS=${STEPS:-3000}
TEACHER=$(latest "$LOGS/b2w_nav_lab_teacher/$TAG")
FUSED=$(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")
LIDAR=$(latest "$LOGS/b2w_nav_lab_lidar_student/$TAG")
EVAL=(python scripts/b2w/evaluate.py --num_envs 1024 --num_steps "$STEPS" --device "$DEVICE" physics=isaacsim_physx)
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab --checkpoint "$TEACHER" > "$OUT/b2w_3_eval_teacher.log" 2>&1
for controller in nominal laggy sluggish weak drifty; do
    "${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --controller "$controller" --map_stats \
        > "$OUT/b2w_3_eval_fused_$controller.log" 2>&1
done
for controller in nominal laggy; do
    "${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --controller "$controller" --shield \
        > "$OUT/b2w_3_eval_fused_${controller}_shield.log" 2>&1
done
for limit in 0.5 1.0; do  # the speed caps of the first trials on the robot (max_speed)
    "${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --speed_limit "$limit" --shield \
        > "$OUT/b2w_3_eval_fused_limit${limit}_shield.log" 2>&1
done
# what the parts of the shield do: without crawling near what moved (crawl:=false on the robot), and the shield alone
# (no crawling, no alternatives, no backing off)
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --shield --shield_set near_moving_gap=0 \
    --out "$(dirname "$FUSED")/eval/b2w_nominal_shield_nocrawl.json" > "$OUT/b2w_3_eval_fused_shield_nocrawl.log" 2>&1
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --shield --shield_set near_moving_gap=0 \
    --guard_set alternatives_below=0,recovery=1e9:0:0 --out "$(dirname "$FUSED")/eval/b2w_nominal_shield_plain.json" \
    > "$OUT/b2w_3_eval_fused_shield_plain.log" 2>&1
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --camera_failure > "$OUT/b2w_3_eval_fused_no_cameras.log" 2>&1
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --lidar_dropout 0.2 --odometry_drift 0.01 \
    --out "$(dirname "$FUSED")/eval/b2w_degraded_map.json" > "$OUT/b2w_3_eval_fused_degraded.log" 2>&1
"${EVAL[@]}" --task G1Patrol-B2W-Navigate-Lab-Lidar --checkpoint "$LIDAR" --map_stats > "$OUT/b2w_3_eval_lidar.log" 2>&1
python scripts/b2w/export.py --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint "$FUSED" --out deploy/b2w/release \
    --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_3_export.log" 2>&1
python scripts/b2w/export.py --task G1Patrol-B2W-Navigate-Lab-Lidar --checkpoint "$LIDAR" --out deploy/b2w/release_lidar \
    --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_3_export_lidar.log" 2>&1
echo "B2-W evaluations: $(dirname "$FUSED")/eval, exported policies: deploy/b2w/release, deploy/b2w/release_lidar"

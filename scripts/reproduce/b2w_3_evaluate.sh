#!/bin/bash
# B2-W stage 3: evaluations on the lab mixture -- the lab teacher; the fused student with the nominal controller, with
# controller stress tests, with the safety shield in the loop (also at the robot's first speed caps, 0.5 and 1 m/s;
# also without crawling near what moved, and the shield alone),
# with failed depth cameras and a degraded map; the lidar student; for the release, the fused student in the civil
# engineering lab (G1Patrol-B2W-Navigate-CivilLab-Fused) -- and the export of both students for the robot
# (deploy/b2w/release: policy.onnx, deploy.json, samples.npz; deploy/b2w/release_lidar). One GPU (B2W_DEVICE, default
# cuda:0), or the evaluations spread over the GPUs of B2W_EVAL_DEVICES (e.g. "cuda:0 cuda:1 cuda:2 cuda:3"; the same
# runs and results).
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
read -r -a DEVICES <<< "${B2W_EVAL_DEVICES:-$DEVICE}"
STEPS=${STEPS:-3000}
RELEASE=${B2W_RELEASE:-deploy/b2w/release}  # where the policies for the robot are exported
TEACHER=$(latest "$LOGS/b2w_nav_lab_teacher/$TAG")
FUSED=$(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")
LIDAR=$(latest "$LOGS/b2w_nav_lab_lidar_student/$TAG")
FUSED_EVAL=$(dirname "$FUSED")/eval
LAB=${B2W_LAB:-Lab}  # the lab task family: Lab (the release), CivilLab (scripts/reproduce/b2w_civil.sh)

# one evaluation per line: log file, then the arguments of scripts/b2w/evaluate.py (the longer ones, with the shield,
# first, so that they spread over the GPUs)
JOBS=(
    "b2w_3_eval_fused_nominal_shield.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller nominal --shield"
    "b2w_3_eval_fused_laggy_shield.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller laggy --shield"
    "b2w_3_eval_fused_limit0.5_shield.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --speed_limit 0.5 --shield"
    "b2w_3_eval_fused_limit1.0_shield.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --speed_limit 1.0 --shield"
    "b2w_3_eval_fused_shield_nocrawl.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --shield --shield_set near_moving_gap=0 --out $FUSED_EVAL/b2w_nominal_shield_nocrawl.json"
    "b2w_3_eval_fused_shield_plain.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --shield --shield_set near_moving_gap=0 --guard_set alternatives_below=0,recovery=1e9:0:0 --out $FUSED_EVAL/b2w_nominal_shield_plain.json"
    "b2w_3_eval_teacher.log --task G1Patrol-B2W-Navigate-$LAB --checkpoint $TEACHER"
    "b2w_3_eval_fused_nominal.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller nominal --map_stats"
    "b2w_3_eval_fused_laggy.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller laggy --map_stats"
    "b2w_3_eval_fused_sluggish.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller sluggish --map_stats"
    "b2w_3_eval_fused_weak.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller weak --map_stats"
    "b2w_3_eval_fused_drifty.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --controller drifty --map_stats"
    "b2w_3_eval_fused_no_cameras.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --camera_failure"
    "b2w_3_eval_fused_degraded.log --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint $FUSED --lidar_dropout 0.2 --odometry_drift 0.01 --out $FUSED_EVAL/b2w_degraded_map.json"
    "b2w_3_eval_lidar.log --task G1Patrol-B2W-Navigate-$LAB-Lidar --checkpoint $LIDAR --map_stats"
)
if [ "$LAB" = Lab ]; then  # the released student in the civil engineering lab (Table B3)
    JOBS+=("b2w_3_eval_fused_civil_mix.log --task G1Patrol-B2W-Navigate-CivilLab-Fused --checkpoint $FUSED --map_stats --out $FUSED_EVAL/b2w_civil_mix.json"
           "b2w_3_eval_fused_civil_mix_shield.log --task G1Patrol-B2W-Navigate-CivilLab-Fused --checkpoint $FUSED --shield --out $FUSED_EVAL/b2w_civil_mix_shield.json")
fi

queue() {  # the jobs i, i + n, i + 2n, ... on device $2 (n = number of devices)
    local i=$1 device=$2 k log
    for ((k = i; k < ${#JOBS[@]}; k += ${#DEVICES[@]})); do
        read -r log args <<< "${JOBS[$k]}"
        # shellcheck disable=SC2086
        python scripts/b2w/evaluate.py --num_envs 1024 --num_steps "$STEPS" --device "$device" $args physics=isaacsim_physx \
            > "$OUT/$log" 2>&1
    done
}
PIDS=()
for i in "${!DEVICES[@]}"; do
    queue "$i" "${DEVICES[$i]}" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "B2-W evaluations failed (see $OUT/b2w_3_eval_*.log)"; exit 1; }
done
python scripts/b2w/export.py --task G1Patrol-B2W-Navigate-$LAB-Fused --checkpoint "$FUSED" --out "$RELEASE" \
    --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_3_export.log" 2>&1
python scripts/b2w/export.py --task G1Patrol-B2W-Navigate-$LAB-Lidar --checkpoint "$LIDAR" --out "${RELEASE}_lidar" \
    --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_3_export_lidar.log" 2>&1
echo "B2-W evaluations: $FUSED_EVAL, exported policies: $RELEASE, ${RELEASE}_lidar"

#!/bin/bash
# B2-W stage 6: the students fine-tuned with their lidar and depth cameras seeing the pedestrians as Isaac Sim
# characters in walking poses (scripts/b2w/bake_people_meshes.py; G1Patrol-B2W-Navigate-Lab-{Fused,Lidar}-People)
# instead of the capsules of stage 2: DAgger with the same teacher, from the students of stage 2, ITERATIONS more
# iterations (default 500; run tag PEOPLE_TAG, default people; B2W_DEVICE and B2W_DEVICE2). Evaluations (on the GPUs
# of B2W_EVAL_DEVICES): both students with people, with people in poses not seen in the fine-tuning (--people_offset),
# with walking people (--walking_people: poses following each pedestrian's gait; also with 2 characters held out) and
# with capsules, the fused one also with the shield, and the students of stage 2 alike. Export to
# B2W_PEOPLE_RELEASE (default deploy/b2w/release_people) and its _lidar, which are not the released policy. Run after
# scripts/reproduce/b2w.sh.
source scripts/reproduce/env.sh
DEVICE=${B2W_DEVICE:-cuda:0}
DEVICE2=${B2W_DEVICE2:-$DEVICE}
read -r -a DEVICES <<< "${B2W_EVAL_DEVICES:-$DEVICE}"
ITERATIONS=${ITERATIONS:-500}
PEOPLE_TAG=${PEOPLE_TAG:-people}
RELEASE=${B2W_PEOPLE_RELEASE:-deploy/b2w/release_people}
if ! ls "${G1_PATROL_PEOPLE_MESHES:-$HOME/.cache/g1_patrol/people_meshes}"/*.npz > /dev/null 2>&1; then
    python scripts/b2w/bake_people_meshes.py > "$OUT/b2w_6_bake.log" 2>&1
fi
finetune() {  # task, checkpoint (a stage-2 student: student, teacher and optimizer), device, log
    isaaclab train --rl_library rsl_rl --external_callback g1_patrol.register_tasks --seed "$SEED" \
        --run_timestamp "$PEOPLE_TAG" --task "$1" --num_envs 4096 --device "$3" --checkpoint "$2" \
        agent.max_iterations="$ITERATIONS" physics=isaacsim_physx > "$OUT/$4" 2>&1
}
finetune G1Patrol-B2W-Navigate-Lab-Fused-People "$(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")" "$DEVICE" \
    b2w_6_fused_people.log &
FUSED_PID=$!
finetune G1Patrol-B2W-Navigate-Lab-Lidar-People "$(latest "$LOGS/b2w_nav_lab_lidar_student/$TAG")" "$DEVICE2" \
    b2w_6_lidar_people.log
wait "$FUSED_PID"
FUSED=$(latest "$LOGS/b2w_nav_lab_fused_student/$PEOPLE_TAG")
LIDAR=$(latest "$LOGS/b2w_nav_lab_lidar_student/$PEOPLE_TAG")
RELEASE_FUSED=$(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")
RELEASE_LIDAR=$(latest "$LOGS/b2w_nav_lab_lidar_student/$TAG")
FUSED_TASK=G1Patrol-B2W-Navigate-Lab-Fused
LIDAR_TASK=G1Patrol-B2W-Navigate-Lab-Lidar
JOBS=(  # the shielded (slow) evaluations first, one per GPU; the students of stage 2 seeing people as the baseline
    "b2w_6_eval_release_fused_people_shield.log --task $FUSED_TASK --checkpoint $RELEASE_FUSED --people_meshes --shield"
    "b2w_6_eval_fused_people_shield.log --task $FUSED_TASK --checkpoint $FUSED --people_meshes --shield"
    "b2w_6_eval_fused_shield.log --task $FUSED_TASK --checkpoint $FUSED --shield"
    "b2w_6_eval_release_fused_people.log --task $FUSED_TASK --checkpoint $RELEASE_FUSED --people_meshes --map_stats"
    "b2w_6_eval_release_lidar_people.log --task $LIDAR_TASK --checkpoint $RELEASE_LIDAR --people_meshes --map_stats"
    "b2w_6_eval_fused_people.log --task $FUSED_TASK --checkpoint $FUSED --people_meshes --map_stats"
    "b2w_6_eval_fused.log --task $FUSED_TASK --checkpoint $FUSED --map_stats"
    "b2w_6_eval_lidar_people.log --task $LIDAR_TASK --checkpoint $LIDAR --people_meshes --map_stats"
    "b2w_6_eval_lidar.log --task $LIDAR_TASK --checkpoint $LIDAR --map_stats"
    "b2w_6_eval_release_fused_people40.log --task $FUSED_TASK --checkpoint $RELEASE_FUSED --people_meshes --people_offset 40"
    "b2w_6_eval_release_lidar_people40.log --task $LIDAR_TASK --checkpoint $RELEASE_LIDAR --people_meshes --people_offset 40"
    "b2w_6_eval_fused_people40.log --task $FUSED_TASK --checkpoint $FUSED --people_meshes --people_offset 40"
    "b2w_6_eval_lidar_people40.log --task $LIDAR_TASK --checkpoint $LIDAR --people_meshes --people_offset 40"
    "b2w_6_eval_release_fused_walking_all.log --task $FUSED_TASK --checkpoint $RELEASE_FUSED --walking_people all"
    "b2w_6_eval_release_fused_walking_held_out.log --task $FUSED_TASK --checkpoint $RELEASE_FUSED --walking_people held_out"
    "b2w_6_eval_fused_walking_all.log --task $FUSED_TASK --checkpoint $FUSED --walking_people all"
    "b2w_6_eval_fused_walking_held_out.log --task $FUSED_TASK --checkpoint $FUSED --walking_people held_out"
    "b2w_6_eval_release_lidar_walking_all.log --task $LIDAR_TASK --checkpoint $RELEASE_LIDAR --walking_people all"
    "b2w_6_eval_lidar_walking_all.log --task $LIDAR_TASK --checkpoint $LIDAR --walking_people all"
)
queue() {  # the jobs i, i + n, ... on device $2
    local i=$1 device=$2 k log args
    for ((k = i; k < ${#JOBS[@]}; k += ${#DEVICES[@]})); do
        read -r log args <<< "${JOBS[$k]}"
        # shellcheck disable=SC2086
        python scripts/b2w/evaluate.py --num_envs 1024 --num_steps 3000 --device "$device" $args \
            physics=isaacsim_physx > "$OUT/$log" 2>&1
    done
}
PIDS=()
for i in "${!DEVICES[@]}"; do
    queue "$i" "${DEVICES[$i]}" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "B2-W evaluations failed (see $OUT/b2w_6_eval_*.log)"; exit 1; }
done
python scripts/b2w/export.py --task G1Patrol-B2W-Navigate-Lab-Fused-People --checkpoint "$FUSED" --out "$RELEASE" \
    --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_6_export.log" 2>&1
python scripts/b2w/export.py --task G1Patrol-B2W-Navigate-Lab-Lidar-People --checkpoint "$LIDAR" \
    --out "${RELEASE}_lidar" --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_6_export_lidar.log" 2>&1
echo "B2-W students seeing people: $FUSED, $LIDAR; exported: $RELEASE, ${RELEASE}_lidar"

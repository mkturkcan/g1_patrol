#!/bin/bash
# Civic stage 4: evaluations of the reaching walker (stress conditions, including attacks, deep squats, a braced
# retreat and hand loads), of the hands controller (scripted triggers) and of the civic teacher and students, written
# to $OUT/eval/<name>.json. Each evaluation runs on one GPU; NUM_GPUS evaluations run at a time.
source scripts/reproduce/env.sh
EVAL="$OUT/eval"
mkdir -p "$EVAL"
WALKER=$(latest "$LOGS/g1_walk_reach/$TAG")
TEACHER=$(latest "$LOGS/g1_civic_teacher/$TAG")
LIDAR=$(latest "$LOGS/g1_civic_lidar_student/$TAG")
CAMERA=$(latest "$LOGS/g1_civic_camera_student/$TAG")
NAV=(--num_envs 2400 --num_steps 1200 physics=isaacsim_physx)
started=0
job() {  # as in 4_evaluate.sh
    local name=$1
    shift
    "$@" --device "cuda:$((started % NUM_GPUS))" --out "$EVAL/$name.json" > "$EVAL/$name.log" 2>&1 &
    started=$((started + 1))
    if ((started % NUM_GPUS == 0)); then wait; fi
}

for c in nominal attacks brace_retreat deep_squat hand_load combined; do
    job "reach_walker_stress_${c}_newton" python scripts/stress_walker.py --walker reach --condition "$c" \
        --checkpoint "$WALKER" --num_envs 1024 --full_terrain
    job "reach_walker_stress_${c}_physx" python scripts/stress_walker.py --walker reach --condition "$c" \
        --checkpoint "$WALKER" --num_envs 1024 --full_terrain physics=isaacsim_physx
done
job hands_litter python scripts/evaluate_hands.py --num_envs 512 physics=isaacsim_physx
job hands_bags python scripts/evaluate_hands.py --num_envs 512 --bags physics=isaacsim_physx
job civic_teacher python scripts/evaluate_nav.py --task G1Patrol-Navigate-Civic --checkpoint "$TEACHER" "${NAV[@]}"
job civic_lidar python scripts/evaluate_nav.py --task G1Patrol-Navigate-Civic-Lidar --checkpoint "$LIDAR" "${NAV[@]}"
job civic_camera python scripts/evaluate_nav.py --task G1Patrol-Navigate-Civic-Camera --checkpoint "$CAMERA" "${NAV[@]}"
for c in attacks incidents robot; do
    job "civic_teacher_stress_$c" python scripts/stress_nav.py --condition "$c" --task G1Patrol-Navigate-Civic \
        --checkpoint "$TEACHER" "${NAV[@]}"
done
for c in attacks sensors robot; do
    job "civic_lidar_stress_$c" python scripts/stress_nav.py --condition "$c" --task G1Patrol-Navigate-Civic-Lidar \
        --checkpoint "$LIDAR" "${NAV[@]}"
    job "civic_camera_stress_$c" python scripts/stress_nav.py --condition "$c" --task G1Patrol-Navigate-Civic-Camera \
        --checkpoint "$CAMERA" "${NAV[@]}"
done
wait
for name in civic_teacher civic_lidar civic_camera; do
    [ -s "$EVAL/$name.json" ] || echo "missing result: $EVAL/$name.json (see $EVAL/$name.log)"
done
echo "civic evaluations: $EVAL"

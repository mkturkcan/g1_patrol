#!/bin/bash
# Stage 4: the evaluations reported in the README, written to $OUT/eval/<name>.json (logs next to them). Each evaluation
# runs on one GPU; NUM_GPUS evaluations run at a time.
source scripts/reproduce/env.sh
EVAL="$OUT/eval"
mkdir -p "$EVAL"
WALKER=$(latest "$LOGS/g1_walk_city/$TAG")
TEACHER=$(latest "$LOGS/g1_nyc_teacher/$TAG")
LIDAR=$(latest "$LOGS/g1_nyc_lidar_student/$TAG")
CAMERA=$(latest "$LOGS/g1_nyc_camera_student/$TAG")
NAV=(--num_envs 2400 --num_steps 600 physics=isaacsim_physx)
started=0
job() {  # job <name> <command...>: run on the next GPU with --out $EVAL/<name>.json, NUM_GPUS at a time
    local name=$1
    shift
    "$@" --device "cuda:$((started % NUM_GPUS))" --out "$EVAL/$name.json" > "$EVAL/$name.log" 2>&1 &
    started=$((started + 1))
    if ((started % NUM_GPUS == 0)); then wait; fi
}

# locomotion: tracking and posture (Newton; navigation command ranges; PhysX), drift, stress in both simulators
job walker_newton python scripts/evaluate.py --task G1Patrol-Walk-City --checkpoint "$WALKER" --num_envs 1024 --full_terrain
job walker_navcmd python scripts/evaluate.py --task G1Patrol-Walk-City --checkpoint "$WALKER" --num_envs 1024 --full_terrain \
    --nav_commands
job walker_physx python scripts/evaluate.py --task G1Patrol-Walk-City --checkpoint "$WALKER" --num_envs 1024 --full_terrain \
    physics=isaacsim_physx
job walker_drift python scripts/evaluate_drift.py --checkpoint "$WALKER" physics=isaacsim_physx
for c in nominal payload weak_motors latency slippery sensor_noise combined; do
    job "walker_stress_${c}_newton" python scripts/stress_walker.py --condition "$c" --checkpoint "$WALKER" --num_envs 1024 \
        --full_terrain
    job "walker_stress_${c}_physx" python scripts/stress_walker.py --condition "$c" --checkpoint "$WALKER" --num_envs 1024 \
        --full_terrain physics=isaacsim_physx
done
# navigation: outcomes, termination analysis of the teacher, degraded conditions
job nav_teacher python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC --checkpoint "$TEACHER" "${NAV[@]}"
job nav_lidar python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC-Lidar --checkpoint "$LIDAR" "${NAV[@]}"
job nav_camera python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC-Camera --checkpoint "$CAMERA" "${NAV[@]}"
job nav_teacher_terminations python scripts/analyze_terminations.py --checkpoint "$TEACHER" \
    --records "$EVAL/nav_teacher_terminations.npz" physics=isaacsim_physx
for c in pedestrians robot; do
    job "nav_teacher_stress_$c" python scripts/stress_nav.py --condition "$c" --task G1Patrol-Navigate-NYC \
        --checkpoint "$TEACHER" "${NAV[@]}"
done
for c in sensors pedestrians robot; do
    job "nav_lidar_stress_$c" python scripts/stress_nav.py --condition "$c" --task G1Patrol-Navigate-NYC-Lidar \
        --checkpoint "$LIDAR" "${NAV[@]}"
    job "nav_camera_stress_$c" python scripts/stress_nav.py --condition "$c" --task G1Patrol-Navigate-NYC-Camera \
        --checkpoint "$CAMERA" "${NAV[@]}"
done
wait
for name in walker_newton walker_navcmd walker_physx walker_drift nav_teacher nav_lidar nav_camera; do
    [ -s "$EVAL/$name.json" ] || echo "missing result: $EVAL/$name.json (see $EVAL/$name.log)"
done
echo "evaluations: $EVAL"

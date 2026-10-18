#!/bin/bash
# Optional: locomotion ablations reported in the README, trained with the stage-1 recipe (walker.sh) and one change
# each (in every phase), and evaluated as the released walker (drift; stress conditions in both simulators):
#   no_drift     without the drift penalty
#   ground_1.0   with Isaac Lab's default ground material (friction 1.0, multiplied) in every phase. MuJoCo-Warp uses
#                the larger of the two coefficients of a contact, so on Newton the contact friction is never below 1.0.
source scripts/reproduce/env.sh
EVAL="$OUT/eval"
mkdir -p "$EVAL"
bash scripts/reproduce/walker.sh "${TAG}_no_drift" env.rewards.velocity_drift=null
bash scripts/reproduce/walker.sh "${TAG}_ground_1.0" \
    env.scene.terrain.physics_material.static_friction=1.0 env.scene.terrain.physics_material.dynamic_friction=1.0 \
    env.scene.terrain.physics_material.friction_combine_mode=multiply env.sim.physics_material.static_friction=1.0 \
    env.sim.physics_material.dynamic_friction=1.0 env.sim.physics_material.friction_combine_mode=multiply
started=0
job() {  # as in 4_evaluate.sh
    local name=$1
    shift
    "$@" --device "cuda:$((started % NUM_GPUS))" --out "$EVAL/$name.json" > "$EVAL/$name.log" 2>&1 &
    started=$((started + 1))
    if ((started % NUM_GPUS == 0)); then wait; fi
}
for name in no_drift ground_1.0; do
    WALKER=$(latest "$LOGS/g1_walk_city/${TAG}_$name")
    job "ablation_${name}_drift" python scripts/evaluate_drift.py --checkpoint "$WALKER" physics=isaacsim_physx
    for c in nominal slippery combined; do
        job "ablation_${name}_stress_${c}_newton" python scripts/stress_walker.py --condition "$c" --checkpoint "$WALKER" \
            --num_envs 1024 --full_terrain
        job "ablation_${name}_stress_${c}_physx" python scripts/stress_walker.py --condition "$c" --checkpoint "$WALKER" \
            --num_envs 1024 --full_terrain physics=isaacsim_physx
    done
done
wait
echo "ablations: $EVAL/ablation_*.json"

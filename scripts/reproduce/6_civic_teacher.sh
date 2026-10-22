#!/bin/bash
# Civic stage 2: navigation teacher of the civic task (litter and trash bags; people who need room, dogs on leashes,
# groups, lines, sidewalk cyclists; someone blocking the way or attacking; someone who has fallen; crosswalk signals)
# with privileged observations, trained from scratch with PPO on PhysX, 16,384 robots on 4 GPUs, in two steps: without
# litter and bags (ITERATIONS_A), then with them from that policy (ITERATIONS). Executes the reaching walker exported by
# stage 1 (src/g1_patrol/assets/data/policies/g1_walk_reach.pt). A step whose run directory holds a .complete marker is
# skipped.
source scripts/reproduce/env.sh
ITERATIONS_A=${ITERATIONS_A:-2500}
ITERATIONS=${ITERATIONS:-3000}
COMMON=(isaaclab train_multigpu --num_gpus "$NUM_GPUS" --external_callback g1_patrol.register_tasks --seed "$SEED"
        --task G1Patrol-Navigate-Civic --num_envs 4096 physics=isaacsim_physx)
STEP_A="$LOGS/g1_civic_teacher/${TAG}_no_litter"
if [ ! -f "$STEP_A/.complete" ]; then
    "${COMMON[@]}" --run_timestamp "${TAG}_no_litter" --master_port 29605 agent.max_iterations="$ITERATIONS_A" \
        "env.commands.nyc.trash_count=[0,0]" "env.commands.nyc.bag_count=[0,0]" > "$OUT/6_civic_teacher_a.log" 2>&1
    touch "$STEP_A/.complete"
fi
"${COMMON[@]}" --run_timestamp "$TAG" --master_port 29608 --checkpoint "$(latest "$STEP_A")" \
    agent.max_iterations="$ITERATIONS" > "$OUT/6_civic_teacher.log" 2>&1
echo "civic teacher: $(latest "$LOGS/g1_civic_teacher/$TAG")"

#!/bin/bash
# Civic stage 2: navigation teacher of the civic task (litter and trash bags; people who need room, dogs on leashes,
# groups, lines, sidewalk cyclists; someone blocking the way or attacking; someone who has fallen; crosswalk signals)
# with privileged observations, trained from scratch with PPO on PhysX, 16,384 robots on 4 GPUs. Executes the reaching
# walker exported by stage 1 (src/g1_patrol/assets/data/policies/g1_walk_reach.pt).
source scripts/reproduce/env.sh
ITERATIONS=${ITERATIONS:-12000}
"${TRAIN[@]}" --master_port 29605 --task G1Patrol-Navigate-Civic --num_envs 4096 agent.max_iterations="$ITERATIONS" \
    physics=isaacsim_physx > "$OUT/6_civic_teacher.log" 2>&1
echo "civic teacher: $(latest "$LOGS/g1_civic_teacher/$TAG")"

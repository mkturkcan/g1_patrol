#!/bin/bash
# Stage 1: locomotion policy (three phases, see walker.sh), exported to src/g1_patrol/assets/data/policies/g1_walk_city.pt,
# which the navigation tasks execute.
source scripts/reproduce/env.sh
bash scripts/reproduce/walker.sh "$TAG"
CHECKPOINT=$(latest "$LOGS/g1_walk_city/$TAG")
python scripts/export_policy.py --task G1Patrol-Walk-City --checkpoint "$CHECKPOINT" \
    --out src/g1_patrol/assets/data/policies/g1_walk_city.pt > "$OUT/1_walker_export.log" 2>&1
echo "walker: $CHECKPOINT -> src/g1_patrol/assets/data/policies/g1_walk_city.pt"

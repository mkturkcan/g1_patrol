#!/bin/bash
# Stage 1: locomotion policy (three phases, see walker.sh), exported to src/g1_patrol/assets/data/policies/<experiment>.pt,
# which the navigation tasks execute: g1_walk_reach.pt (default) or, with WALKER_TASK=G1Patrol-Walk-City, the walking-only
# g1_walk_city.pt of the first release.
source scripts/reproduce/env.sh
export WALKER_TASK=${WALKER_TASK:-G1Patrol-Walk-Reach}
case "$WALKER_TASK" in
    G1Patrol-Walk-Reach) EXP=g1_walk_reach ;;
    G1Patrol-Walk-City) EXP=g1_walk_city ;;
    *) echo "unknown walker task $WALKER_TASK"; exit 1 ;;
esac
bash scripts/reproduce/walker.sh "$TAG"
CHECKPOINT=$(latest "$LOGS/$EXP/$TAG")
python scripts/export_policy.py --task "$WALKER_TASK" --checkpoint "$CHECKPOINT" \
    --out "src/g1_patrol/assets/data/policies/$EXP.pt" > "$OUT/1_walker_export_$EXP.log" 2>&1
echo "walker: $CHECKPOINT -> src/g1_patrol/assets/data/policies/$EXP.pt"

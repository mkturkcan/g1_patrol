#!/bin/bash
# Reproduce the civic release (litter and trash bags, people who need room, dogs, groups, lines, cyclists, attacks,
# people blocking the way, someone who has fallen, crosswalk signals) and its evaluations: reaching walker, civic
# teacher, students, evaluations. Run from the repository root with the Isaac Lab environment active.
set -euo pipefail
bash scripts/reproduce/1_walker.sh
bash scripts/reproduce/6_civic_teacher.sh
bash scripts/reproduce/7_civic_students.sh
bash scripts/reproduce/8_civic_evaluate.sh

#!/bin/bash
# Reproduce the first release (street navigation) and its evaluations: walking-only walker, teacher, students,
# evaluations (about 25 h on four NVIDIA RTX PRO 6000 GPUs). Run from the repository root with the Isaac Lab
# environment active. The civic release is reproduced by civic.sh.
set -euo pipefail
WALKER_TASK=G1Patrol-Walk-City bash scripts/reproduce/1_walker.sh
bash scripts/reproduce/2_teacher.sh
bash scripts/reproduce/3_students.sh
bash scripts/reproduce/4_evaluate.sh

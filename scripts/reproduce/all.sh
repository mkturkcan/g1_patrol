#!/bin/bash
# Reproduce the released policies and their evaluations: walker, teacher, students, evaluations (about 25 h on four
# NVIDIA RTX PRO 6000 GPUs). Run from the repository root with the Isaac Lab environment active.
set -euo pipefail
bash scripts/reproduce/1_walker.sh
bash scripts/reproduce/2_teacher.sh
bash scripts/reproduce/3_students.sh
bash scripts/reproduce/4_evaluate.sh

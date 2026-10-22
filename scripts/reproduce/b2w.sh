#!/bin/bash
# Reproduce the B2-W release: navigation teacher, lidar student, evaluations and the export for the robot. Run from
# the repository root with the Isaac Lab environment active; B2W_DEVICE selects the GPU (default cuda:0).
set -euo pipefail
bash scripts/reproduce/b2w_1_teacher.sh
bash scripts/reproduce/b2w_2_student.sh
bash scripts/reproduce/b2w_3_evaluate.sh

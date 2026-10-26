#!/bin/bash
# The civil engineering lab experiment (README, Table B3): the lab teacher fine-tuned and the students distilled on the
# mixture in which the civil engineering lab takes half of the lab tiles (G1Patrol-B2W-Navigate-CivilLab*), starting
# from the general teacher of the release (run B2W_TEACHER_TAG, default release; scripts/reproduce/b2w_1_teacher.sh);
# evaluations; export to deploy/b2w/release_civil, which is not the released policy. Run tag civil (TAG). Run from the
# repository root with the Isaac Lab environment active, after scripts/reproduce/b2w.sh (or its first two steps).
set -euo pipefail
export B2W_LAB=CivilLab TAG=${TAG:-civil} B2W_TEACHER_TAG=${B2W_TEACHER_TAG:-release}
export B2W_RELEASE=${B2W_RELEASE:-deploy/b2w/release_civil}
bash scripts/reproduce/b2w_1b_lab_teacher.sh
bash scripts/reproduce/b2w_2_student.sh
bash scripts/reproduce/b2w_3_evaluate.sh

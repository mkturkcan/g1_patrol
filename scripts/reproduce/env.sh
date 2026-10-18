# Settings shared by the reproduction scripts. Source from the repository root with the Isaac Lab environment active.
#   NUM_GPUS   GPUs per training run (the released policies: 4)
#   SEED       base seed; each GPU of a run uses SEED + its rank
#   TAG        name of the run directories, logs/rsl_rl/<experiment>/<TAG>
set -euo pipefail
NUM_GPUS=${NUM_GPUS:-4}
SEED=${SEED:-42}
TAG=${TAG:-release}
LOGS=logs/rsl_rl
OUT=${OUT:-logs/reproduce}
mkdir -p "$OUT"
TRAIN=(isaaclab train_multigpu --num_gpus "$NUM_GPUS" --external_callback g1_patrol.register_tasks --seed "$SEED"
       --run_timestamp "$TAG")

# latest checkpoint of a run directory
latest() { ls "$1"/model_*.pt | sort -V | tail -1; }

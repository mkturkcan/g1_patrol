#!/bin/bash
# B2-W stage 5: the post-training alignment of the camera student's encoder. Replays of the camera environment, driven
# by the fused student (scripts/b2w/camera_replays.py, two seeds; B2W_CAMERA_DEVICE, default cuda:0), their transfer to
# realistic video with NVIDIA Cosmos 3 Nano, one look per replay (scripts/b2w/camera_transfer.py, on the GPUs of
# B2W_TRANSFER_DEVICES; it runs with COSMOS_PYTHON, the python of an environment with diffusers' Cosmos 3 pipelines, and
# COSMOS3_NANO, a local download of nvidia/Cosmos3-Nano), the fine-tuning of the last blocks of the encoder on them
# (scripts/b2w/camera_align.py): <camera student run>/alignment/encoder, report.json; and the export of the camera
# student with the aligned encoder (scripts/b2w/export_camera.py; B2W_CAMERA_RELEASE, default
# deploy/b2w/release_camera). Run after b2w_4_camera.sh.
source scripts/reproduce/env.sh
DEVICE=${B2W_CAMERA_DEVICE:-cuda:0}
read -r -a DEVICES <<< "${B2W_TRANSFER_DEVICES:-$DEVICE}"
: "${COSMOS_PYTHON:?the python of an environment with the diffusers Cosmos 3 pipelines}"
: "${COSMOS3_NANO:?a local download of nvidia/Cosmos3-Nano}"
FUSED=$(latest "$LOGS/b2w_nav_lab_fused_student/$TAG")
CAMERA=$(latest "$LOGS/b2w_nav_lab_camera_student/$TAG")
ALIGN=$(dirname "$CAMERA")/alignment
for seed in 42 7; do
    python scripts/b2w/camera_replays.py --checkpoint "$FUSED" --num_envs 32 --clips 2 --seed "$seed" \
        --out "$ALIGN/replays_$seed" --device "$DEVICE" physics=isaacsim_physx > "$OUT/b2w_5_replays_$seed.log" 2>&1
done
PIDS=()
for i in "${!DEVICES[@]}"; do
    "$COSMOS_PYTHON" scripts/b2w/camera_transfer.py --replays "$ALIGN/replays_42" "$ALIGN/replays_7" \
        --out "$ALIGN/transfers" --model "$COSMOS3_NANO" --looks --device "${DEVICES[$i]}" --worker "$i" \
        --workers "${#DEVICES[@]}" > "$OUT/b2w_5_transfer_$i.log" 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "B2-W transfers failed (see $OUT/b2w_5_transfer_*.log)"; exit 1; }
done
python scripts/b2w/camera_align.py --transfers "$ALIGN/transfers" --policy "$CAMERA" --out "$ALIGN" --device "$DEVICE" \
    > "$OUT/b2w_5_align.log" 2>&1
python scripts/b2w/export_camera.py --checkpoint "$CAMERA" --encoder "$ALIGN/encoder" --replays "$ALIGN/replays_42" \
    --out "${B2W_CAMERA_RELEASE:-deploy/b2w/release_camera}" > "$OUT/b2w_5_export.log" 2>&1
echo "B2-W camera student's aligned encoder: $ALIGN/encoder ($ALIGN/report.json)"
echo "B2-W camera student's export: ${B2W_CAMERA_RELEASE:-deploy/b2w/release_camera}"

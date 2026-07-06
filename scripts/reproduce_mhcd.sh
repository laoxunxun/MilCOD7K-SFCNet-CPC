#!/usr/bin/env bash
# Reproduce the cross-dataset evaluation on MHCD2022 (Table 10).
#
# MHCD2022 is a bounding-box dataset. We convert SFCNet-CPC's per-class
# segmentation mask to detection boxes with a SINGLE FIXED configuration
# (binarize threshold 0.5, min component area 100 px, IoU>=0.5), identical
# for all categories — no per-category or test-set threshold tuning — so the
# zero-shot cross-dataset evaluation is fair and reproducible.
# (person<->soldier, military_vehicle<->vehicle, tank<->tank;
#  aeroplane/warship excluded — no model channel.)
set -e
cd "$(dirname "$0")/.."

CKPT=${CKPT:-checkpoints/sfcnet_cpc_t015.pth}
MHCD=${MHCD:-data/MHCD2022}

# 1) (optional) exclude the ~1 image that overlaps the MilCOD7K training set
#    python tools/detect_mhcd_overlap.py --milcod data/MilCOD7K --mhcd "$MHCD" \
#                                       --out data/MHCD2022_clean

# 2) fixed-threshold, channel-matched evaluation -> Table 10
python tools/eval_mhcd_cross.py \
    --checkpoint "$CKPT" \
    --mhcd-img  "$MHCD/JPEGImages" \
    --mhcd-ann  "$MHCD/Annotations" \
    --mhcd-split "$MHCD/ImageSets/Main/test.txt" \
    --thresh 0.5 --min-area 100 --iou 0.5 \
    --out results/mhcd_table10.json

echo "Expected F1 (fixed-threshold, fair): tank 0.54 > person 0.43 > military-vehicle 0.41."
echo "(Precision 0.38-0.58, Recall 0.41-0.51 — moderate cross-dataset transfer.)"

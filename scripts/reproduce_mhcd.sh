#!/usr/bin/env bash
# Reproduce the cross-dataset evaluation on MHCD2022.
#
# MHCD2022 is a bounding-box dataset, so this needs (a) an overlap check
# against the MilCOD7K training set and (b) a box-matched, channel-matched
# evaluator. Those scripts (detect_mhcd_overlap.py, evaluate_mhcd_box.py)
# live in the original workspace and are NOT bundled in this minimal release.
#
# To reproduce end-to-end, copy them here from the source workspace, then run:
set -e
cd "$(dirname "$0")/.."

# 1) exclude the ~1 image that overlaps the MilCOD7K training set
# python detect_mhcd_overlap.py --milcod data/MilCOD7K --mhcd data/MHCD2022 \
#                              --out data/MHCD2022_clean

# 2) channel-matched evaluation (person/tank/vehicle; plane & warship excluded)
# python evaluate_mhcd_box.py \
#     --checkpoint cpts_sfcnet_cpc_t015/Net_multi_best_iou.pth \
#     --mhcd data/MHCD2022_clean --out results/mhcd_table10.csv

echo "NOTE: detect_mhcd_overlap.py and evaluate_mhcd_box.py are not bundled."
echo "Copy them from the original workspace and uncomment the lines above."
echo "Expected F1: tank 0.63 > vehicle 0.57 > person 0.50."

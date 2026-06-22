#!/usr/bin/env bash
# Reproduce the ablation rows that this train.py can express (ablation study).
# Hyperparameters come from the YAML; per-row knobs use real CLI flags.
#
# NOTE — rows NOT expressible from this minimal train.py:
#   * baseline (SFCNet, no CPC) and "C: refine-only (no contrast)" need a
#     no-CPC / contrast-off code path. Train them with the baseline scripts in
#     the original workspace, or extend models/ accordingly.
set -e
cd "$(dirname "$0")/.."

# D: contrast only (refinement OFF)            -> 82.10
python train.py --config configs/sfcnet_cpc_t015.yaml --enable_refinement False --save_path ./cpts_ablation_D/
# CPC default τ=0.07                           -> 82.85
python train.py --config configs/sfcnet_cpc_t007.yaml
# E1 / E2: λ scan (0.05 / 0.20)
python train.py --config configs/sfcnet_cpc_t007.yaml --cpc_weight 0.05 --save_path ./cpts_E1/
python train.py --config configs/sfcnet_cpc_t007.yaml --cpc_weight 0.20 --save_path ./cpts_E2/
# F1: τ=0.03 ; F2: τ=0.15 (recommended, 84.93)
python train.py --config configs/sfcnet_cpc_t015.yaml --temperature 0.03  --save_path ./cpts_F1/
python train.py --config configs/sfcnet_cpc_t015.yaml
# Variants
python train.py --config configs/sfcnet_fscpc.yaml
python train.py --config configs/sfcnet_cacpc.yaml

echo "Done. Per-class IoU is saved under each cpts_*/training_history.json."

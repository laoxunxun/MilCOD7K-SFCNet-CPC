#!/usr/bin/env bash
# Reproduce our model from the main comparison.
set -e
cd "$(dirname "$0")/.."

echo "=== Our model: SFCNet-CPC (τ=0.15) -> 84.93 mIoU / 91.71 mF1 ==="
python train.py --config configs/sfcnet_cpc_t015.yaml

echo "=== Evaluate on the test set ==="
python test.py --config configs/sfcnet_cpc_t015.yaml \
               --checkpoint cpts_sfcnet_cpc_t015/Net_multi_best_iou.pth

# ------------------------------------------------------------------
# NOTE: the comparison baselines (DeepLabV3+, U-Net, PSPNet, FPN, PAN,
# FPNet, SINet-V2, ZoomNeXt) and the binary COD metrics are
# produced by separate baseline train/eval scripts that live in the
# original workspace (baselines/, FPNet/, SINet-V2-main/, ZoomNeXt-main/).
# They are not bundled in this minimal release. Bring them over if you
# need to regenerate Tables 4–6 end to end.
# ------------------------------------------------------------------
echo "Done."

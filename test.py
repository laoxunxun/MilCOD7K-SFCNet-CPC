"""SFCNet-CPC multi-class evaluation on the MilCOD7K test set.

Computes per-class IoU / F1 and mIoU / mF1 (original-resolution protocol:
prediction upsampled back to the original GT resolution, argmax over 5 classes).

Usage:
    python test.py --config configs/sfcnet_cpc_t015.yaml \
                   --checkpoint cpts_sfcnet_cpc_t015/Net_multi_best_iou.pth

    # or without a config:
    python test.py --checkpoint cpts_sfcnet_cpc_t015/Net_multi_best_iou.pth \
                   --data_root data/MilCOD7K --use_fscpc/--use_cccpc as needed
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

sys.path.append('./models')
sys.path.append('.')
from models.NetMultiClass import NetMultiClass
from models.NetFS_CPC import NetFS_CPC
from models.NetCC_CPC import NetCC_CPC
from data_multiclass import test_dataset_multiclass

CLASS_NAMES = ['background', 'camouflage_soldier', 'military_vehicle', 'tank', 'fortification']


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise ImportError('--config requires PyYAML: pip install pyyaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def build_model(cfg, args):
    """Build the model corresponding to the config (or CLI flags)."""
    m = cfg.get('model', {}) or {}
    cpc = cfg.get('cpc', {}) or {}
    fscpc = cfg.get('fscpc', {}) or {}
    cacpc = cfg.get('cacpc', {}) or {}

    name = m.get('name', 'NetMultiClass')
    if args.use_fscpc or name == 'NetFS_CPC':
        name = 'NetFS_CPC'
    if args.use_cccpc or name == 'NetCC_CPC':
        name = 'NetCC_CPC'

    temp = cpc.get('temperature') or cacpc.get('base_temperature') or fscpc.get('hf_temperature') or args.temperature
    refine = cpc.get('use_refinement', args.enable_refinement)
    nc = args.num_classes or m.get('num_classes') or 5

    if name == 'NetFS_CPC':
        return NetFS_CPC(num_classes=nc, temperature=temp, enable_refinement=refine,
                         hf_temperature=fscpc.get('hf_temperature'), lf_temperature=fscpc.get('lf_temperature')), nc
    if name == 'NetCC_CPC':
        return NetCC_CPC(num_classes=nc, temperature=temp, enable_refinement=refine), nc
    return NetMultiClass(num_classes=nc, temperature=temp, enable_refinement=refine), nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--checkpoint', type=str, required=True, help='model weights .pth')
    ap.add_argument('--data_root', type=str, default=None, help='dataset root (contains test/)')
    ap.add_argument('--testsize', type=int, default=None)
    ap.add_argument('--num_classes', type=int, default=None)
    ap.add_argument('--gpu_id', type=str, default='0')
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--enable_refinement', type=lambda x: str(x).lower() == 'true', default=True)
    ap.add_argument('--use_fscpc', action='store_true')
    ap.add_argument('--use_cccpc', action='store_true')
    ap.add_argument('--save_json', type=str, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    data_root = args.data_root or (cfg.get('data', {}) or {}).get('root') or 'data/MilCOD7K'
    testsize = args.testsize or (cfg.get('data', {}) or {}).get('train_size') or 384

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    model, num_classes = build_model(cfg, args)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_root = os.path.join(data_root, 'test')
    loader = test_dataset_multiclass(os.path.join(test_root, 'Imgs'), testsize, num_classes)
    gt_root = os.path.join(test_root, 'GT')

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    print(f"Evaluating {loader.size} test images at {testsize}px...")
    with torch.no_grad():
        for _ in range(loader.size):
            image, name = loader.load_data()
            stem = os.path.splitext(name)[0]
            gt = np.load(os.path.join(gt_root, stem + '.npy'))     # H×W×C
            H, W = gt.shape[:2]

            out = model(image.to(device))
            out = out[0] if isinstance(out, tuple) else out         # (1,C,h,w)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)  # H×W
            gt_lbl = gt.argmax(2).astype(np.uint8) if gt.ndim == 3 else gt.astype(np.uint8)

            for c in range(num_classes):
                tp[c] += np.logical_and(pred == c, gt_lbl == c).sum()
                fp[c] += np.logical_and(pred == c, gt_lbl != c).sum()
                fn[c] += np.logical_and(pred != c, gt_lbl == c).sum()

    iou = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)

    names = CLASS_NAMES[:num_classes] if num_classes <= len(CLASS_NAMES) else [f'class_{i}' for i in range(num_classes)]
    print("\n%-26s %8s %8s %8s" % ("class", "IoU", "F1", "support"))
    for i, n in enumerate(names):
        print("%-26s %7.2f%% %7.2f%% %8d" % (n, iou[i] * 100, f1[i] * 100, int(tp[i] + fn[i])))
    print("-" * 54)
    print("%-26s %7.2f%% %7.2f%%" % ("mIoU / mF1", iou.mean() * 100, f1.mean() * 100))

    if args.save_json:
        result = {n: {'iou': float(iou[i]), 'f1': float(f1[i])} for i, n in enumerate(names)}
        result['mIoU'] = float(iou.mean())
        result['mF1'] = float(f1.mean())
        with open(args.save_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {args.save_json}")


if __name__ == '__main__':
    main()

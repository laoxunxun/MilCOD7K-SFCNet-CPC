#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-dataset evaluation on MHCD2022 (mask -> box, fixed-threshold, fair).

SFCNet-CPC is a segmentation network; MHCD2022 has bounding-box annotations.
This script converts each class channel's probability map to detection boxes
with a SINGLE FIXED configuration (no per-category / test-set tuning), so the
cross-dataset (zero-shot) evaluation is fair and reproducible.

Rule (identical for all categories):
  channel prob map -> binarize at threshold (default 0.5) -> external contours
  -> bounding box of each component (drop area < min_area, default 100 px)
  -> greedy match to GT boxes by IoU >= 0.5 -> Precision / Recall / F1.

Category matching:  person <-> camouflaged_soldier, military_vehicle <-> military_vehicle,
tank <-> tank.  aeroplane / warship are excluded (no model channel).

Usage:
  python tools/eval_mhcd_cross.py \
      --checkpoint cpts_sfcnet_cpc_t015/Net_multi_best_iou.pth \
      --mhcd-img /path/MHCD2022/JPEGImages \
      --mhcd-ann /path/MHCD2022/Annotations \
      --mhcd-split /path/MHCD2022/ImageSets/Main/test.txt
"""
import os, sys, json, argparse, numpy as np, cv2, torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.NetMultiClass import NetMultiClass
import torchvision.transforms as T

CATS = {'person': (1, 'camouflaged_soldier'),
        'military vehicle': (2, 'military_vehicle'),
        'tank': (3, 'tank')}


def parse_xml(p):
    root = ET.parse(p).getroot(); boxes = {k: [] for k in CATS}
    for o in root.findall('object'):
        nm = o.find('name').text.strip()
        if nm in CATS and o.find('bndbox') is not None:
            b = o.find('bndbox')
            boxes[nm].append([int(float(b.find(t).text)) for t in ['xmin', 'ymin', 'xmax', 'ymax']])
    return boxes


def mask_to_boxes(mask_u8, thresh, min_area):
    _, bw = cv2.threshold(mask_u8, int(thresh * 255), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            out.append([x, y, x + w, y + h])
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def match(preds, gts, iou_th):
    used = [False] * len(gts); tp = fp = 0
    for p in preds:
        best, bi = -1, 0.0
        for i, g in enumerate(gts):
            v = iou(p, g)
            if v > bi and not used[i]:
                bi, best = v, i
        if best >= 0 and bi >= iou_th:
            used[best] = True; tp += 1
        else:
            fp += 1
    return tp, fp, len(gts) - sum(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--mhcd-img', required=True)
    ap.add_argument('--mhcd-ann', required=True)
    ap.add_argument('--mhcd-split', required=True)
    ap.add_argument('--num-classes', type=int, default=5)
    ap.add_argument('--thresh', type=float, default=0.5, help='fixed binarization threshold (no tuning)')
    ap.add_argument('--min-area', type=int, default=100, help='min component area (px), noise filter')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--out', default='results/mhcd_table10.json')
    args = ap.parse_args()

    names = [l.strip() for l in open(args.mhcd_split) if l.strip()]
    print(f'MHCD test: {len(names)} imgs | FIXED thresh={args.thresh} min_area={args.min_area} IoU={args.iou}')
    m = NetMultiClass(num_classes=args.num_classes, temperature=0.07, enable_refinement=True)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m.load_state_dict(sd, strict=False); m = m.cuda().eval()
    tr = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                    T.Normalize([.485, .456, .406], [.229, .224, .225])])
    agg = {k: [0, 0, 0] for k in CATS}
    for n in names:
        img = Image.open(os.path.join(args.mhcd_img, n + '.jpg')).convert('RGB'); W, H = img.size
        gts = parse_xml(os.path.join(args.mhcd_ann, n + '.xml'))
        with torch.no_grad():
            out = m(tr(img).unsqueeze(0).cuda())
            logits = out if not isinstance(out, (list, tuple)) else [o for o in out if o is not None][-1]
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            probs = torch.softmax(logits, 1).squeeze(0).cpu().numpy()
        for cat, (idx, _) in CATS.items():
            tp, fp, fn = match(mask_to_boxes((probs[idx] * 255).astype(np.uint8), args.thresh, args.min_area),
                               gts[cat], args.iou)
            a = agg[cat]; a[0] += tp; a[1] += fp; a[2] += fn
    res = {}
    print('\n=== Cross-dataset (fixed-threshold, fair) ===')
    for cat, (tp, fp, fn) in agg.items():
        P = tp / (tp + fp + 1e-9); R = tp / (tp + fn + 1e-9); F1 = 2 * P * R / (P + R + 1e-9)
        res[cat] = {'Precision': P, 'Recall': R, 'F1': F1, 'GT': tp + fn, 'TP': tp, 'FP': fp, 'FN': fn}
        print(f'  {cat:16s} GT={tp+fn:4d}  P={P:.3f}  R={R:.3f}  F1={F1:.3f}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump({'config': vars(args), 'per_category': res}, open(args.out, 'w'), indent=2)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
# Reproduce the AI-generated-image ablation (real-only / ai-only / mixed).
#
# This train.py trains on whatever is under data_root, so to get the
# real-only / ai-only subsets we first materialise filtered data dirs
# (symlinks) using metadata.csv, then train on each.
set -e
cd "$(dirname "$0")/.."

DATA=data/MilCOD7K          # full dataset (Format A)
SUB=data/MilCOD7K_sub       # filtered subsets will be built here

# 1) build real-only / ai-only subsets (symlinks pointing back at the full data)
python - <<'PY'
import os, csv
root='data/MilCOD7K'; out='data/MilCOD7K_sub'
rows=list(csv.DictReader(open('metadata.csv',encoding='utf-8')))
for mode in ['real_only','ai_only']:
    keep={'real_only':'real','ai_only':'ai'}[mode]
    for split in ['train','val','test']:
        for kind in ['Imgs','GT','Edge']:
            os.makedirs(f'{out}/{mode}/{split}/{kind}',exist_ok=True)
    names={r['filename'] for r in rows if r['source']==keep}
    for r in rows:
        if r['source']!=keep: continue
        stem=os.path.splitext(r['filename'])[0]
        split=r['split']
        for kind,ext in [('Imgs','.jpg'),('GT','.npy'),('Edge','.png')]:
            src=f'{root}/{split}/{kind}/{stem}{ext}'
            dst=f'{out}/{mode}/{split}/{kind}/{stem}{ext}'
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src),dst)
    print(f'{mode}: {len(names)} images linked')
PY

# 2) train SFCNet-F2 (τ=0.15) on each subset, eval on the FULL test set
python train.py --config configs/sfcnet_cpc_t015.yaml --data_root $SUB/real_only --save_path ./cpts_ai_real_only/
python train.py --config configs/sfcnet_cpc_t015.yaml --data_root $SUB/ai_only   --save_path ./cpts_ai_ai_only/
python train.py --config configs/sfcnet_cpc_t015.yaml                             # mixed == main result (84.93)

echo "Evaluate all three on the FULL test set to fill the real/AI comparison."
echo "(real-only 43.33 / ai-only 32.13 / mixed 84.93 mIoU)"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert MilCOD7K **Format A** (GT/*.npy, H x W x C one-hot) into **Format B**
(single-channel class-index PNG, pixel = class id) — the standard semantic-
segmentation layout (PASCAL/Cityscapes/MMSeg friendly).

Example
-------
python tools/npy_to_classpng.py \
        --in  data/MilCOD7K \
        --out data/MilCOD7K_std \
        --copy-images
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def npy_to_class_index(npy_path: str) -> np.ndarray:
    """H x W x C one-hot uint8  ->  H x W class-index uint8 (argmax)."""
    arr = np.load(npy_path)
    if arr.ndim == 3:                 # (H, W, C) one-hot
        idx = arr.argmax(axis=2).astype(np.uint8)
    elif arr.ndim == 2:               # already a class-index map
        idx = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected GT shape {arr.shape} in {npy_path}")
    return idx


def convert_split(in_root: Path, out_root: Path, split: str, copy_images: bool):
    in_split = in_root / split
    gt_dir = in_split / "GT"
    img_dir = in_split / "Imgs"
    out_img = out_root / split / "images"
    out_msk = out_root / split / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(gt_dir.glob("*.npy"))
    for npy in tqdm(npy_files, desc=f"{split}"):
        idx = npy_to_class_index(str(npy))
        Image.fromarray(idx, mode="L").save(out_msk / (npy.stem + ".png"))

        if copy_images:
            # match by stem (image extension may be .jpg)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_dir / (npy.stem + ext)
                if cand.exists():
                    img_path = cand
                    break
            if img_path is not None:
                shutil.copy2(img_path, out_img / img_path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Format-A root (has train/val/test/{Imgs,GT,Edge})")
    ap.add_argument("--out", dest="out", required=True, help="output Format-B root")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--copy-images", action="store_true", help="also copy Imgs/ -> images/")
    args = ap.parse_args()

    in_root, out_root = Path(args.inp), Path(args.out)
    for s in args.splits:
        convert_split(in_root, out_root, s, args.copy_images)
    print(f"Done. Class-index masks written to {out_root} (0=bg,1=soldier,2=vehicle,3=tank,4=fortification).")


if __name__ == "__main__":
    main()

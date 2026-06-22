#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate single-channel boundary **edge maps** from class-index masks
(either Format-A GT/*.npy or Format-B masks/*.png). Edges = morphological
gradient of the class map; saved as 0/255 PNGs matching the native Edge/.

Example
-------
python tools/make_edge_maps.py \
        --masks data/MilCOD7K_std/train/masks \
        --out   data/MilCOD7K/train/Edge \
        --mode  png            # or 'npy' if masks are .npy one-hot
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def class_index_from(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        return arr.argmax(axis=2).astype(np.uint8) if arr.ndim == 3 else arr.astype(np.uint8)
    return np.array(Image.open(path).convert("L"))


def edges_from_classmap(cls: np.ndarray, ksize: int = 3) -> np.ndarray:
    dil = cv2.dilate(cls, np.ones((ksize, ksize), np.uint8), iterations=1)
    ero = cv2.erode(cls, np.ones((ksize, ksize), np.uint8), iterations=1)
    edge = (dil != ero).astype(np.uint8) * 255
    return edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="dir of class-index masks (.png) or GT (.npy)")
    ap.add_argument("--out", required=True, help="output Edge/ dir")
    ap.add_argument("--mode", choices=["png", "npy"], default="png")
    ap.add_argument("--ksize", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ext = "*.npy" if args.mode == "npy" else "*.png"
    files = sorted(Path(args.masks).glob(ext))
    for f in tqdm(files, desc="edges"):
        cls = class_index_from(f)
        edge = edges_from_classmap(cls, args.ksize)
        Image.fromarray(edge, mode="L").save(out / (f.stem + ".png"))
    print(f"Done. Edge maps -> {out}")


if __name__ == "__main__":
    main()

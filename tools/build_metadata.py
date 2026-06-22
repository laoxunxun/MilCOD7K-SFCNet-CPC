#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build metadata.csv for MilCOD7K.

Real/AI rule (matches the expected counts — train 3986/1774, val 420/300,
test 538/182):
    _mcod followed by EXACTLY 5 digits  ->  real     (zero-padded, e.g. mcod00159)
    _mcod followed by <5 digits         ->  ai       (e.g. mcod1196)

The 'i_' filename prefix marks interference images (an independent dimension);
it is recorded in a separate column and does not affect the real/ai label.

Output columns: filename, split, class, class_id, source, interference
    class_id: 0=bg 1=soldier 2=vehicle 3=tank 4=fortification (matches the model)
    source:   'real' | 'ai'
"""
import argparse
import csv
import os
import re

PREFIX2CLS = {"cs": ("camouflage_soldier", 1),
              "mv": ("military_vehicle", 2),
              "t":  ("tank", 3),
              "f":  ("fortification", 4)}


def classify(filename: str):
    """Return (class_name, class_id, source, interference) or None if unrecognised."""
    name = filename[:-4] if filename.lower().endswith(".jpg") else filename
    interference = name.startswith("i_")
    core = name[2:] if interference else name
    pfx = core.split("_", 1)[0]
    if pfx not in PREFIX2CLS:
        return None
    cls_name, cls_id = PREFIX2CLS[pfx]
    m = re.search(r"_mcod(\d+)\.", filename) or re.search(r"_mcod(\d+)$", name)
    source = "real" if (m and len(m.group(1)) == 5) else "ai"
    return cls_name, cls_id, source, interference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/MilCOD7K",
                    help="Format-A root with {train,val,test}/Imgs")
    ap.add_argument("--out", default="metadata.csv")
    args = ap.parse_args()

    rows, counts = [], {"real": 0, "ai": 0}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(args.data_root, split, "Imgs")
        if not os.path.isdir(img_dir):
            print(f"[skip] {img_dir} not found")
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(".jpg"):
                continue
            info = classify(fn)
            if info is None:
                continue
            cls_name, cls_id, source, interf = info
            rows.append([fn, split, cls_name, cls_id, source, interf])
            counts[source] += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "class", "class_id", "source", "interference"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"  real: {counts['real']}   ai: {counts['ai']}")
    print("Reference counts:  train 3986/1774, val 420/300, test 538/182")


if __name__ == "__main__":
    main()

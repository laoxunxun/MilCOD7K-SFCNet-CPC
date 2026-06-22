# Where to put the data

This folder is gitignored — the MilCOD7K dataset is **not** stored in git.
Download the native archive (see [`../docs/dataset.md`](../docs/dataset.md)) and unpack here so the tree looks like:

```
data/
└── MilCOD7K/                         # native format — what our loader reads
    ├── train/{Imgs,GT,Edge}/
    ├── val/{Imgs,GT,Edge}/
    └── test/{Imgs,GT,Edge}/
```

If you only have another layout (class-index PNG / YOLOseg), convert it to the native one with the scripts in [`../tools/`](../tools/) — see `../docs/dataset.md`.

Verify counts:
```bash
for s in train val test; do
  echo "$s Imgs=$(ls data/MilCOD7K/$s/Imgs | wc -l) GT=$(ls data/MilCOD7K/$s/GT | wc -l)"
done
# expect: train 5760/5760  val 720/720  test 720/720
```

> Tip: a tiny `sample/` ships in the repo for format inspection.

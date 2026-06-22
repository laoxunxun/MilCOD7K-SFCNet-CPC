# Sample (format inspection)

A handful of MilCOD7K images (one per class + two `i_`-prefixed interference
examples) so you can inspect the data layout without downloading the full set:

```
sample/
├── Imgs/   *.jpg        # cs_=soldier, mv_=vehicle, t_=tank, f_=fortification
├── GT/     *.npy        # H x W x 5 one-hot masks (uint8, 0/1)
└── Edge/   *.png        # boundary maps
```

This is for **format inspection** only — it is not a runnable train/val/test
split (there are only 6 images, no split folders). To run training or evaluation,
download the full MilCOD7K into `data/` and then:

```bash
# quick 1-epoch sanity run on the full data
python train.py --config configs/sfcnet_cpc_t015.yaml --epoch 1
```

You can also use the sample to try the converters, e.g. turn a one-hot `.npy`
into a class-index PNG:
```bash
python tools/npy_to_classpng.py --in sample --out /tmp/sample_std
```

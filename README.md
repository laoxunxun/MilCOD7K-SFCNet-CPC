# MilCOD7K · SFCNet-CPC

**Multi-Class Camouflaged Military Object Detection via Spatial–Frequency Collaborative Network with Category Prototype Contrast**

A PyTorch project for multi-class camouflaged military object segmentation. It builds on the third-party **SFCNet** backbone (Spatial–Frequency Collaborative Network, [Zhao et al., TMM 2025]) and adds the **MilCOD7K dataset**, the **Category Prototype Contrast (CPC)** module, and the **FS-CPC / CA-CPC** variants.

---

## 📌 Overview

- **MilCOD7K** — a multi-class military camouflage segmentation benchmark: 7,200 pixel-level annotated images (4 camouflaged classes + background), mixing real and AI-generated samples.
- **CPC module** — learnable class prototypes supervised by an InfoNCE objective, with a prototypical refinement path, to encourage inter-class separability and intra-class compactness.
- **Temperature finding** — with only a few class prototypes, a *higher* contrastive temperature (τ=0.15) tends to act like label smoothing and helps discrimination, which differs from the low-temperature convention of instance-level contrastive learning.
- **Results** — 84.93% mIoU / 91.71% mF1 on MilCOD7K, higher than the general segmentation and COD-specific baselines evaluated under the same protocol.

## 📊 Main Results (MilCOD7K test set)

| Method | mIoU (%) ↑ | mF1 (%) ↑ | Params (M) ↓ |
|---|---|---|---|
| PSPNet | 60.17 | 73.75 | 24.32 |
| FPN | 76.13 | 86.07 | 26.12 |
| PAN | 76.53 | 86.35 | 24.26 |
| U-Net | 77.35 | 86.85 | 32.52 |
| DeepLabV3+ | 79.54 | 88.36 | 26.68 |
| FPNet (COD) | 62.90 | 75.48 | 75.60 |
| ZoomNeXt (COD) | 59.37 | 73.06 | 28.50 |
| SINet-V2 (COD) | 22.29 | 26.24 | 27.00 |
| SAM2-DEGNet (COD, 5-class adapted †) | 83.21 | 90.65 | 11.80 trainable / 224 total |
| SFCNet (backbone, no CPC) | 80.76 | 89.08 | 23.32 |
| **SFCNet-CPC (τ=0.15, Ours)** | **84.93** | **91.71** | **23.32** |

† SAM2-DEGNet is a binary COD method; this row reports its 5-class head adaptation (trained 15 epochs under a limited compute budget). On the binary sub-problem it is the strongest method (bIoU 0.868); see `docs/reproduce.md`.

**Complexity (GFLOPs = MACs, COD convention):** SFCNet-CPC needs 24.28 GFLOPs / 23.32M params (all trained) — far lighter in compute than SAM2-DEGNet (195.60 GFLOPs, ~3× slower) and FPNet (38.90 GFLOPs / 75.6M). The CPC module adds only +0.7% GFLOPs and zero extra parameters over the SFCNet baseline (24.12 → 24.28).

Full per-class breakdown, ablations, binary COD metrics, cross-dataset (MHCD2022) results, and repeated-run variance are in [`docs/reproduce.md`](docs/reproduce.md). **Dataset construction, sources, annotation, and ethics are documented in [`DATASET.md`](DATASET.md).**

---

## 📁 Repository Structure

```
MilCOD7K-SFCNet-CPC/
├── README.md
├── LICENSE                      # code: MIT · data: CC-BY-NC 4.0
├── CITATION.cff
├── requirements.txt
├── train.py                     # main training entry (SFCNet-CPC)
├── train_multiscale.py          # multi-scale training variant
├── test.py                      # evaluation
├── infer.py                     # inference / visualization
├── config.py  options.py  lr_scheduler.py  optimizer.py
├── data_multiclass.py           # dataset loader (reads *.jpg / *.npy / *.png)
├── models/                      # SFCNet + CPC + variants + baselines
│   ├── model.py                 # SFCNet core (DWT, FAI, …) + CategoryPrototypeContrast
│   ├── smt.py                   # SMT-Tiny backbone
│   ├── NetMultiClass.py         # SFCNet-CPC  (our main model)
│   ├── NetFS_CPC.py             # FS-CPC variant
│   ├── NetCC_CPC.py             # CA-CPC variant
│   ├── Net.py  SINetV2MultiClass.py
├── pytorch_losses/              # CE / Tversky / Focal / CPC losses
├── configs/                     # YAML configs for each model/variant
├── tools/                       # dataset conversion utilities
│   ├── convert_yolo_to_multiclass.py   # YOLOseg  -> native format (npy)
│   ├── npy_to_classpng.py              # npy one-hot -> class-index PNG (community std)
│   └── make_edge_maps.py               # generate boundary edge maps
├── scripts/                     # one-command train/eval/reproduce shells
├── docs/                        # dataset / method / reproduction docs
├── data/                        # (gitignored) put downloaded MilCOD7K here
├── sample/                      # tiny sample (5 imgs/class) for quick smoke test
└── checkpoints/                 # (gitignored) pretrained weights (Baidu Netdisk, see below)
```

---

## 🛠 Installation

```bash
git clone https://github.com/laoxunxun/MilCOD7K-SFCNet-CPC.git
cd MilCOD7K-SFCNet-CPC
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, single NVIDIA RTX 3090 (24 GB). SMT-Tiny backbone is initialized from ImageNet-1K pretrained weights (download instructions in [`docs/reproduce.md`](docs/reproduce.md)).

---

## 📦 Dataset & Weights (Baidu Netdisk)

The dataset and pretrained weights are distributed via **Baidu Netdisk** (百度网盘);
they are not stored in git. Download and place the dataset under `data/` and the
weights under `checkpoints/`.

**MilCOD7K dataset + pretrained weights** are in the same Baidu Netdisk share
(dataset in native format — see [`docs/dataset.md`](docs/dataset.md); weights — see
[`checkpoints/README.md`](checkpoints/README.md)):
```
链接: https://pan.baidu.com/s/1-nO8Ty_jNAjL6pOg5-oPVg?pwd=ufkn
提取码: ufkn
```
Unpack the dataset under `data/` and put the weights under `checkpoints/`. Verify
integrity against the `SHA256SUMS.txt` provided in the same share.
The native layout once unpacked:
```
data/MilCOD7K/{train,val,test}/{Imgs,GT,Edge}/
```
- `Imgs/*.jpg`, `GT/*.npy` (H×W×5 one-hot), `Edge/*.png` — exactly what our dataloader reads.

**Classes (index : name):** `0:background · 1:camouflage_soldier · 2:military_vehicle · 3:tank · 4:fortification`

**Splits:** train 5,760 / val 720 / test 720 (8:1:1). Real photos: 4,944 · AI-generated: 2,256. A `metadata.csv` flags each image as `real` or `ai` and its class, so the real/AI ablation is reproducible.

Need another layout (class-index PNG or YOLOseg)? Convert with the scripts in [`tools/`](tools/) — see [`docs/dataset.md`](docs/dataset.md).

---

## 🚀 Quick Start

### 1. Train SFCNet-CPC (recommended config, τ=0.15)
```bash
python train.py --config configs/sfcnet_cpc_t015.yaml
```

### 2. Evaluate
```bash
python test.py --config configs/sfcnet_cpc_t015.yaml \
               --checkpoint checkpoints/sfcnet_cpc_t015.pth
```

### 3. Inference on a single image / folder
```bash
python infer.py --checkpoint checkpoints/sfcnet_cpc_t015.pth \
                --input path/to/image_or_dir --output results/
```

### 4. Smoke test (no download needed)
A tiny sample ships under `sample/` (5 images per class) so you can verify the pipeline end-to-end in seconds.

See [`scripts/`](scripts/) for one-command recipes and [`docs/reproduce.md`](docs/reproduce.md) to reproduce every reported result.

---

## 🔁 Reproduce the Results

| Target | Script |
|---|---|
| Main comparison | `scripts/reproduce_main.sh` |
| Ablations | `scripts/reproduce_ablation.sh` |
| Real/AI-image impact | `scripts/reproduce_ai_ablation.sh` |
| Cross-dataset on MHCD2022 | `scripts/reproduce_mhcd.sh` |

---

## 📜 License

- **Code:** [MIT License](LICENSE).
- **MilCOD7K dataset:** released under **CC BY-NC 4.0** for **non-commercial research only**. The dataset contains real photographs collected from public sources / existing datasets and AI-generated images; see [`docs/dataset.md`](docs/dataset.md) for provenance and terms.
- **SFCNet backbone:** belongs to its original authors ([Zhao et al., TMM 2025]); please cite their work.

## 🙏 Acknowledgements

- SFCNet backbone: Zhao et al., "Spatial-Frequency Collaborative Learning for Camouflaged Object Detection," *IEEE TMM*, 2025.
- Baselines: DeepLabV3+, U-Net, PSPNet, FPN, PAN, FPNet, SINet-V2, ZoomNeXt (each in its original release).
- AI image generation: CamDiff (Camouflage Image Augmentation via Diffusion, *CAAI AIR* 2023), built on the RunwayML Stable Diffusion Inpainting model; public text-to-image services Doubao and Nano-banana-1.
- Annotation: labelme.
- Evaluation: py_sod_metrics for the standard COD metrics.

## 📖 Citation

If you find this project useful, please cite the repository:

```bibtex
@software{milcod7k_sfcnet_cpc,
  author = {Haoran Sun, Jue Qu, Haiping Liu, Wei Wang},
  title  = {{MilCOD7K-SFCNet-CPC}: Multi-Class Camouflaged Military Object Segmentation},
  url    = {https://github.com/laoxunxun/MilCOD7K-SFCNet-CPC},
  year   = {2026}
}
```

See [`CITATION.cff`](CITATION.cff).

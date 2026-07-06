# Reproducing the Results

Hardware used: **single NVIDIA RTX 3090 (24 GB)**, PyTorch 2.x, input 384×384, batch 12, Adam (lr 5e-5, wd 1e-4), 200 epochs max, ReduceLROnPlateau + early stop (patience 20).

## 0. Prepare data & backbone init

1. Download MilCOD7K (Format A) into `data/` — see [`dataset.md`](dataset.md). Expected layout:
   ```
   data/MilCOD7K/{train,val,test}/{Imgs,GT,Edge}/
   ```
2. Download the **SMT-Tiny ImageNet-1K** pretrained weights (used to initialize the encoder) and place at `checkpoints/smt_tiny_imagenet1k.pth` (or set the path in the config). Source: official SMT release.

## 1. Main comparison

```bash
# our model (recommended config)
bash scripts/reproduce_main.sh
```
This trains/evaluates SFCNet-CPC (τ=0.15) and the baselines (DeepLabV3+, U-Net, PSPNet, FPN, PAN, FPNet, SINet-V2, ZoomNeXt) under the **same** CE+Tversky+Focal protocol, then prints the comparison.

Expected headline: **SFCNet-F2 = 84.93 mIoU / 91.71 mF1**.

## 2. Ablations

```bash
bash scripts/reproduce_ablation.sh
```
Runs every ablation row: baseline (no CPC), C (refine only), D (contrast only), CPC default (τ=0.07), E1/E2 (λ scan), F1/F2 (τ scan), FS-CPC, CA-CPC. Per-class IoU is dumped to `results/ablation_perclass.csv`.

## 3. AI-generated image impact

```bash
bash scripts/reproduce_ai_ablation.sh
```
Trains SFCNet-F2 on `real-only`, `ai-only`, and `mixed` (filter by `metadata.csv`). Expected: mixed 84.93 ≫ real-only 43.33 ≫ ai-only 32.13.

## 4. Cross-dataset on MHCD2022

```bash
bash scripts/reproduce_mhcd.sh
```
Applies the MilCOD7K-trained model directly to **MHCD2022** (no fine-tuning), channel-matched evaluation. First run `python tools/detect_mhcd_overlap.py` to exclude the 1 overlapping image. The mask→box conversion uses a **single fixed configuration** (binarize threshold 0.5, min component area 100 px, IoU ≥ 0.5) for all categories — no per-category or test-set threshold tuning — so the zero-shot cross-dataset evaluation is fair. Expected (Table 10): F1 tank 0.54 > person 0.43 > military-vehicle 0.41 (Precision 0.38–0.58, Recall 0.41–0.51 — moderate transfer).

## Pretrained checkpoints

Trained weights for the main models are provided via Baidu Netdisk (see [`checkpoints/README.md`](../checkpoints/README.md)):

| Checkpoint | mIoU | mF1 |
|---|---|---|
| `sfcnet_cpc_t015.pth` (SFCNet-F2, **highest mIoU**) | 84.93 | 91.71 |
| `sfcnet_cpc_t007.pth` (default) | 82.85 | 90.44 |
| `sfcnet_fscpc.pth` | 84.24 | 91.31 |
| `sfcnet_baseline.pth` (no CPC) | 80.76 | 89.08 |

## Tips / caveats

- **Run-to-run variance:** across 3 random seeds SFCNet-CPC (τ=0.15) varies by only **±0.14 mIoU** (82.60 ± 0.14%) and ±0.09 mF1 (90.26 ± 0.09%) on the MilCOD7K test set — the method is highly seed-stable. (Trained under a reduced compute budget; the main-table 84.93% uses the full protocol.)
- **FPS:** our model runs ~29.7 FPS at 384² on a 3090 — slower than lightweight baselines (PSPNet ~95) due to multi-level DWT + windowed cross-attention.
- If you only want to verify the pipeline without downloading the full dataset, use the bundled `sample/` (format inspection).

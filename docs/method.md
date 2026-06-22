# Method Overview

This note summarizes the method so users of the code know what each file does.

## Backbone: SFCNet (third-party, [Zhao et al., TMM 2025])

We adopt SFCNet's spatial–frequency collaborative architecture as the backbone
(**not our contribution**). Its core pieces live in [`models/model.py`](../models/model.py):

- `smt_t` ([`models/smt.py`](../models/smt.py)) — SMT-Tiny ImageNet-1K pretrained encoder, four hierarchical stages `[64,128,256,512]`.
- `DWT` — Discrete Wavelet Transform decodes each stage into low-frequency semantics `LL` and high-frequency textures `LH/HL/HH`.
- `FAI` (Frequency Adaptive Integration) — windowed cross-attention with **HF as Query, LF as Key/Value**, sharpening high-frequency detail at boundaries and attending to low-frequency semantics in smooth regions.
- `UpSampler`, decoder dense aggregation, refinement conv → 5-class pixel-wise head.

## Our Contribution 1 — CPC module

`CategoryPrototypeContrast` in [`models/model.py`](../models/model.py), assembled in [`models/NetMultiClass.py`](../models/NetMultiClass.py).

- Maintains **5 learnable prototype vectors** (one per class, 64-d), updated by gradient descent.
- **InfoNCE contrastive loss** pulls each pixel's feature toward its true-class prototype and away from the others (jointly enforces inter-class separability + intra-class compactness).
- **Prototypical refinement path** uses softmax-normalized prototype similarities as attention weights to recalibrate decoder features before the classifier.

> Ablation finding: the refinement branch *alone* hurts (78.18 vs 80.76 baseline) — it only helps *after* contrastive learning has organized the prototype space. Keep them together.

## Our Contribution 2 — Variants (FS-CPC, CA-CPC)

- **FS-CPC** ([`models/NetFS_CPC.py`](../models/NetFS_CPC.py)) — maintains **separate prototype banks for the HF and LF streams**, combined via a lightweight spatial **frequency gate** `g`. Adds cross-frequency alignment + gate-diversity regularization.
- **CA-CPC** ([`models/NetCC_CPC.py`](../models/NetCC_CPC.py)) — confusion-aware: an **EMA confusion matrix** up-weights easily-confused class pairs in the InfoNCE loss, a **boundary-adaptive temperature** sharpens predictions near edges, and a **prototype-separation loss** keeps confusable prototypes apart. Introduces no new learnable parameters.

> Note: in our 5-class setting the standard CPC gives the highest mIoU among the three. FS-CPC is close; CA-CPC's separation loss stays inactive (margin 0.3 < min prototype distance ~0.78) and trails the others. These are best regarded as **diagnosed exploratory variants** rather than strict improvements.

## The temperature insight (key takeaway)

CPC performance **increases monotonically** with temperature: τ = 0.03 → 0.07 → 0.15 gives 80.11 → 82.85 → **84.93** mIoU. This contradicts the instance-level contrastive-learning convention ("lower τ = sharper gradients").

**Why:** with only 5 prototypes, very low τ makes the softmax near one-hot, so non-top-1 gradients vanish and ambiguous-boundary pixels can't be corrected. A moderate τ (0.15, ~4.56% of max entropy) acts like **label smoothing** — confident yet producing useful gradients for all classes. At τ ≥ 0.5 the distribution collapses toward uniform and the signal fails.

**Use τ = 0.15** (`configs/sfcnet_cpc_t015.yaml`) — this is our recommended configuration (the one with the highest measured mIoU).

## Loss

Total objective (see [`pytorch_losses/`](../pytorch_losses/) and the loss block in `train.py`):

```
L = L_CE(class-weighted) + L_Tversky(α=0.7, β=0.3) + L_Focal + λ·L_CPC + Σ deep-supervision aux
```

Class weights counter the 88.45% background imbalance; Tversky penalizes missed detections; Focal focuses on hard pixels; CPC provides feature-level supervision; deep supervision comes from intermediate decoder stages.

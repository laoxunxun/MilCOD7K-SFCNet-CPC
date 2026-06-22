# Pretrained checkpoints

Weights are distributed via **Baidu Netdisk** (百度网盘), not stored in git —
download and place them here.

```
链接: https://pan.baidu.com/s/1-nO8Ty_jNAjL6pOg5-oPVg?pwd=ufkn
提取码: ufkn
```

| File | Model | mIoU | mF1 | Config |
|---|---|---|---|---|
| `smt_tiny_imagenet1k.pth` | SMT-Tiny encoder (ImageNet-1K) | — | — | encoder init |
| `sfcnet_cpc_t015.pth` | **SFCNet-CPC (τ=0.15), highest mIoU** | 84.93 | 91.71 | `configs/sfcnet_cpc_t015.yaml` |
| `sfcnet_cpc_t007.pth` | SFCNet-CPC (τ=0.07) | 82.85 | 90.44 | `configs/sfcnet_cpc_t007.yaml` |
| `sfcnet_fscpc.pth` | FS-CPC variant | 84.24 | 91.31 | `configs/sfcnet_fscpc.yaml` |
| `sfcnet_cacpc.pth` | CA-CPC variant | 80.61 | 89.02 | `configs/sfcnet_cacpc.yaml` |
| `sfcnet_baseline.pth` | SFCNet, no CPC | 80.76 | 89.08 | — |

After downloading, verify each file against the bundled `SHA256SUMS.txt`:
```bash
sha256sum -c SHA256SUMS.txt
```

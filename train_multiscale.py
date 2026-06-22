# -*- coding: utf-8 -*-
"""
Multi-scale CPC training script
Supported config:
  - multi_scale: True (multi-scale CPC) / False (original single-scale CPC)
  - adaptive_temp: True (adaptive temperature) / False (shared temperature)
  - shared_prototypes: True (shared prototypes) / False (independent prototypes)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import json
import torch.backends.cudnn as cudnn
import sys

sys.path.append('./models')
sys.path.append('.')
from models.NetMultiScaleCPC import NetMultiScaleCPC
from data_multiclass import SalObjDatasetMultiClass, get_loader_multiclass


# Loss functions (same as original SFCNet)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, ignore_bg=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        if self.ignore_bg:
            pred = pred[:, 1:, :]
            target = target[:, 1:, :]
        tp = (pred * target).sum(dim=2)
        fn = ((1 - target) * pred).sum(dim=2)
        fp = (target * (1 - pred)).sum(dim=2)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        target_labels = torch.argmax(target, dim=1)
        ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target_labels)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(CombinedLoss, self).__init__()
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, pred, target):
        target_labels = torch.argmax(target, dim=1)
        if hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(pred.device)
        ce = self.ce_loss(pred, target_labels)
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return ce + tversky + 0.3 * focal, ce, tversky, focal


class MetricsAccumulator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, pred, target):
        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
        target_labels = torch.argmax(target, dim=1).cpu().numpy()
        for cls in range(self.num_classes):
            self.tp[cls] += ((pred_labels == cls) & (target_labels == cls)).sum()
            self.fp[cls] += ((pred_labels == cls) & (target_labels != cls)).sum()
            self.fn[cls] += ((pred_labels != cls) & (target_labels == cls)).sum()

    def compute_metrics(self):
        metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        for cls in range(self.num_classes):
            tp, fp, fn = self.tp[cls], self.fp[cls], self.fn[cls]
            iou = tp / (tp + fp + fn + 1e-6)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            metrics['iou'].append(iou)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
        metrics['mean_iou'] = np.mean(metrics['iou'])
        metrics['mean_f1'] = np.mean(metrics['f1'])
        return metrics


def train_epoch(train_loader, model, criterion, optimizer, device, num_classes, cpc_weight=0.1):
    model.train()
    total_loss = 0
    total_cpc = 0
    metrics_accum = MetricsAccumulator(num_classes)

    for i, (images, gts, edges) in enumerate(train_loader):
        images = Variable(images).to(device, non_blocking=True)
        gts = Variable(gts).to(device, non_blocking=True)
        gt_labels = torch.argmax(gts, dim=1)

        optimizer.zero_grad()
        model_outputs = model(images, gt_labels=gt_labels)
        outputs = model_outputs[0]
        cpc_loss = model_outputs[-1] if isinstance(model_outputs, tuple) and len(model_outputs) >= 7 else None

        loss, ce, tversky, focal = criterion(outputs, gts)

        # Deep supervision
        ds_weights = [0.4, 0.3, 0.2, 0.1]
        for idx, ds_w in enumerate(ds_weights):
            aux_out = model_outputs[idx + 1]
            aux_loss, _, _, _ = criterion(aux_out, gts)
            loss = loss + ds_w * aux_loss

        if cpc_loss is not None:
            loss = loss + cpc_weight * cpc_loss
            total_cpc += cpc_loss.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            metrics_accum.update(outputs, gts)

        if (i + 1) % 100 == 0:
            cpc_str = f', CPC: {cpc_loss.item():.4f}' if cpc_loss is not None else ''
            print(f'    Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}{cpc_str}', flush=True)

        del images, gts, gt_labels, outputs, model_outputs, loss

    train_metrics = metrics_accum.compute_metrics()
    return {'loss': total_loss / len(train_loader), 'cpc': total_cpc / len(train_loader), 'iou': train_metrics['mean_iou']}


def validate(val_loader, model, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    metrics_accum = MetricsAccumulator(num_classes)

    with torch.no_grad():
        for images, gts, edges in val_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            model_outputs = model(images)
            outputs = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            loss, _, _, _ = criterion(outputs, gts)
            total_loss += loss.item()
            metrics_accum.update(outputs, gts)
            del images, gts, outputs, model_outputs, loss

    return {'loss': total_loss / len(val_loader), 'metrics': metrics_accum.compute_metrics()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--adaptive_temp', action='store_true', help='Use adaptive temperature per class')
    parser.add_argument('--shared_prototypes', action='store_true', help='Share prototypes across scales')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--cpc_weight', type=float, default=0.1)

    # Data
    parser.add_argument('--rgb_root', type=str, default='./Dataset_multiclass_5class_new/train/Imgs/')
    parser.add_argument('--gt_root', type=str, default='./Dataset_multiclass_5class_new/train/GT/')
    parser.add_argument('--edge_root', type=str, default='./Dataset_multiclass_5class_new/train/Edge/')
    parser.add_argument('--val_rgb_root', type=str, default='./Dataset_multiclass_5class_new/val/Imgs/')
    parser.add_argument('--val_gt_root', type=str, default='./Dataset_multiclass_5class_new/val/GT/')
    parser.add_argument('--val_edge_root', type=str, default='./Dataset_multiclass_5class_new/val/Edge/')

    # Training
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1.0, 3.0, 2.0, 3.0, 2.0])
    parser.add_argument('--load_pretrained', type=str, default=None)

    # Save
    parser.add_argument('--save_path', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-generate save path
    if args.save_path is None:
        suffix = 'ms_cpc'
        if args.adaptive_temp:
            suffix += '_atemp'
        if args.shared_prototypes:
            suffix += '_shared'
        args.save_path = f'./cpts_{suffix}/'

    class_names = ['background', 'camouflage_soldier', 'military_vehicle', 'tank', 'fortification']

    print("=" * 70)
    print("Multi-Scale CPC Training")
    print(f"  Adaptive Temp: {args.adaptive_temp}")
    print(f"  Shared Prototypes: {args.shared_prototypes}")
    print(f"  Temperature: {args.temperature}")
    print(f"  CPC Weight: {args.cpc_weight}")
    print("=" * 70)

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # Model
    model = NetMultiScaleCPC(
        num_classes=args.num_classes,
        temperature=args.temperature,
        enable_refinement=True,
        adaptive_temp=args.adaptive_temp,
        shared_prototypes=args.shared_prototypes
    )

    if args.load_pretrained:
        print(f"\nLoading pretrained weights: {args.load_pretrained}")
        state_dict = torch.load(args.load_pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print('Weights loaded (strict=False)')

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParams: {total_params:,} ({total_params/1e6:.2f}M)")

    # Data
    print("\nLoading data...")
    train_loader = get_loader_multiclass(
        args.rgb_root, args.gt_root, args.edge_root,
        batchsize=args.batch_size, trainsize=args.trainsize,
        num_classes=args.num_classes, shuffle=True, num_workers=0, augmentation=True
    )
    val_loader = get_loader_multiclass(
        args.val_rgb_root, args.val_gt_root, args.val_edge_root,
        batchsize=args.batch_size, trainsize=args.trainsize,
        num_classes=args.num_classes, shuffle=False, num_workers=0, augmentation=False
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Loss & Optimizer
    criterion = CombinedLoss(args.num_classes, class_weights=args.class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Resume
    start_epoch = 0
    best_val_iou = 0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'lr': [], 'per_class_iou': []}

    checkpoint_path = os.path.join(args.save_path, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        start_epoch = ckpt['epoch']
        best_val_iou = ckpt.get('best_val_iou', 0)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        history = ckpt.get('history', history)
        print(f"Resumed from epoch {start_epoch}")

    patience_counter = 0

    print("\nTraining...")
    for epoch in range(start_epoch, args.epoch):
        epoch_start = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_stats = train_epoch(train_loader, model, criterion, optimizer, device, args.num_classes, cpc_weight=args.cpc_weight)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_stats = validate(val_loader, model, criterion, device, args.num_classes)

        scheduler.step(val_stats['loss'])
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['train_iou'].append(train_stats['iou'])
        history['val_iou'].append(val_stats['metrics']['mean_iou'])
        history['lr'].append(current_lr)
        history['per_class_iou'].append(val_stats['metrics']['iou'])

        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\nEpoch [{epoch+1:03d}/{args.epoch}] ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_stats['loss']:.4f}, IoU: {train_stats['iou']:.4f}, CPC: {train_stats['cpc']:.4f}")
        print(f"  Val   - Loss: {val_stats['loss']:.4f}, IoU: {val_stats['metrics']['mean_iou']:.4f}")
        for i, name in enumerate(class_names):
            print(f"    {name}: {val_stats['metrics']['iou'][i]:.4f}")

        if val_stats['metrics']['mean_iou'] > best_val_iou:
            best_val_iou = val_stats['metrics']['mean_iou']
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            patience_counter = 0
            print(f"  -> Best IoU: {best_val_iou:.4f}")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'history': history
            }, checkpoint_path)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Print adaptive temperatures if used
    if args.adaptive_temp:
        print("\n=== Learned Adaptive Temperatures ===")
        for i, name in enumerate(class_names):
            cpc = model.cpc_f1 if not args.shared_prototypes else model.cpc_f1
            temps = cpc.get_temperatures()
            print(f"  {name}: {temps[i].item():.4f}")

    print("\n" + "=" * 70)
    print(f"Training complete! Best val IoU: {best_val_iou:.4f}")
    print(f"Params: {total_params:,} ({total_params/1e6:.2f}M)")
    print("=" * 70)

    with open(os.path.join(args.save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Summary
    print("\n=== RESULTS ===")
    print(f"Config: multi_scale=True, adaptive_temp={args.adaptive_temp}, shared_proto={args.shared_prototypes}")
    final_metrics = history['per_class_iou'][-1] if history['per_class_iou'] else [0]*args.num_classes
    for i, name in enumerate(class_names):
        print(f"  {name}: IoU={final_metrics[i]:.2f}%")
    print(f"  mIoU: {best_val_iou*100:.2f}%")


if __name__ == '__main__':
    main()

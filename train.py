"""
SFCNet multi-class segmentation training script (improved v3 - two-stage training)
- pretrained weight loading (single-class backbone -> multi-class fine-tuning)
- early stopping (monitor val_IoU)
- combined loss (CE + Tversky + Focal)
- Deep Supervision (F2-F5 auxiliary losses)
- class-weight balancing
- full evaluation metrics
- GPU optimization (cudnn.benchmark)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
import numpy as np
from datetime import datetime
import json
import torch.backends.cudnn as cudnn

sys.path.append('./models')
sys.path.append('.')
from models.NetMultiClass import NetMultiClass
from models.NetFS_CPC import NetFS_CPC
from models.NetCC_CPC import NetCC_CPC
from data_multiclass import SalObjDatasetMultiClass, get_loader_multiclass
import pytorch_losses as losses


class EarlyStopping:
    """Early stopping - monitors val_IoU (higher is better)"""
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_iou, model, save_path):
        score = val_iou  # higher IoU is better

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
                print(f'  current val IoU: {val_iou:.4f}, best: {self.best_score:.4f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'  val IoU improved: {self.best_score:.4f} -> {score:.4f}')
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_multi_best.pth'))
            print(f'  -> saved best-Loss model')


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation (excluding background)"""
    def __init__(self, smooth=1e-6, ignore_bg=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        if self.ignore_bg:
            pred = pred[:, 1:, :]
            target = target[:, 1:, :]
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss (alpha controls the FN penalty, beta controls the FP penalty)"""
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
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - logits
        target: (B, C, H, W) - one-hot encoded
        """
        # Convert one-hot to class indices
        target_labels = torch.argmax(target, dim=1)  # (B, H, W)

        ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target_labels)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss: CE + Tversky + Focal (with optional class weights)"""
    def __init__(self, num_classes, ce_weight=1.0, tversky_weight=1.0, focal_weight=0.3, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes

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

        total_loss = (self.ce_weight * ce +
                     self.tversky_weight * tversky +
                     self.focal_weight * focal)
        return total_loss, ce, tversky, focal


class MetricsAccumulator:
    """Accumulate evaluation metric statistics online"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, pred, target):
        """Update the statistics"""
        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
        target_labels = torch.argmax(target, dim=1).cpu().numpy()

        for cls in range(self.num_classes):
            self.tp[cls] += ((pred_labels == cls) & (target_labels == cls)).sum()
            self.fp[cls] += ((pred_labels == cls) & (target_labels != cls)).sum()
            self.fn[cls] += ((pred_labels != cls) & (target_labels == cls)).sum()

    def compute_metrics(self):
        """Compute metrics from accumulated statistics"""
        metrics = {
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for cls in range(self.num_classes):
            tp = self.tp[cls]
            fp = self.fp[cls]
            fn = self.fn[cls]

            iou = tp / (tp + fp + fn + 1e-6)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            metrics['iou'].append(iou)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)

        metrics['mean_iou'] = np.mean(metrics['iou'])
        metrics['mean_precision'] = np.mean(metrics['precision'])
        metrics['mean_recall'] = np.mean(metrics['recall'])
        metrics['mean_f1'] = np.mean(metrics['f1'])
        metrics['per_class_iou'] = metrics['iou']
        metrics['per_class_f1'] = metrics['f1']

        return metrics


def calculate_metrics(pred, target, num_classes):
    """Compute metrics: IoU, Precision, Recall, F1"""
    # Get predictions
    pred_labels = torch.argmax(pred, dim=1)  # (B, H, W)
    target_labels = torch.argmax(target, dim=1)  # (B, H, W)

    metrics = {
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((pred_labels == cls) & (target_labels == cls)).sum().item()
        fp = ((pred_labels == cls) & (target_labels != cls)).sum().item()
        fn = ((pred_labels != cls) & (target_labels == cls)).sum().item()

        # IoU
        iou = tp / (tp + fp + fn + 1e-6)
        metrics['iou'].append(iou)

        # Precision and Recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        metrics['f1'].append(f1)

    # Calculate mean
    metrics['mean_iou'] = np.mean(metrics['iou'])
    metrics['mean_precision'] = np.mean(metrics['precision'])
    metrics['mean_recall'] = np.mean(metrics['recall'])
    metrics['mean_f1'] = np.mean(metrics['f1'])

    # Per-class metrics
    metrics['per_class_iou'] = metrics['iou']
    metrics['per_class_f1'] = metrics['f1']

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--num_classes', type=int, default=5,
                        help='number of classes (including background)')
    parser.add_argument('--rgb_root', type=str,
                        default='./Dataset_multiclass_5class_new/train/Imgs/',
                        help='training image directory')
    parser.add_argument('--gt_root', type=str,
                        default='./Dataset_multiclass_5class_new/train/GT/',
                        help='training GT directory')
    parser.add_argument('--edge_root', type=str,
                        default='./Dataset_multiclass_5class_new/train/Edge/',
                        help='training edge directory')
    parser.add_argument('--val_rgb_root', type=str,
                        default='./Dataset_multiclass_5class_new/val/Imgs/',
                        help='validation image directory')
    parser.add_argument('--val_gt_root', type=str,
                        default='./Dataset_multiclass_5class_new/val/GT/',
                        help='validation GT directory')
    parser.add_argument('--val_edge_root', type=str,
                        default='./Dataset_multiclass_5class_new/val/Edge/',
                        help='validation edge directory')

    # training parameters
    parser.add_argument('--epoch', type=int, default=200,
                        help='max training epochs')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch size')
    parser.add_argument('--trainsize', type=int, default=384,
                        help='training image size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID')

    # config file (recommended: python train.py --config configs/sfcnet_cpc_t015.yaml)
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file; its values override the defaults (explicit CLI flags still take precedence)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='dataset root (contains train/val/test/{Imgs,GT,Edge}); auto-sets the *_root paths')

    # early-stopping parameters
    parser.add_argument('--patience', type=int, default=20,
                        help='early-stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='early-stopping min-improvement threshold')

    # loss weights
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='CE loss weight')
    parser.add_argument('--tversky_weight', type=float, default=1.0,
                        help='Tversky loss weight')
    parser.add_argument('--focal_weight', type=float, default=0.3,
                        help='Focal loss weight')
    parser.add_argument('--cpc_weight', type=float, default=0.1,
                        help='CPC contrastive loss weight')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='CPC softmax temperature')
    parser.add_argument('--hf_temperature', type=float, default=None,
                        help='FS-CPC HF-branch temperature (None = use temperature)')
    parser.add_argument('--lf_temperature', type=float, default=None,
                        help='FS-CPC LF-branch temperature (None = use temperature)')
    parser.add_argument('--enable_refinement', type=lambda x: x.lower() == 'true',
                        default=True,
                        help='enable CPC feature refinement (True/False)')
    parser.add_argument('--use_fscpc', action='store_true',
                        help='use FS-CPC (frequency-sensitive prototype contrast) instead of standard CPC')
    parser.add_argument('--use_cccpc', action='store_true',
                        help='use CC-CPC (confusion-aware boundary-enhanced prototype contrast) instead of standard CPC')
    parser.add_argument('--class_weights', type=float, nargs='+',
                        default=[1.0, 3.0, 2.0, 3.0, 2.0],
                        help='class-weight list (for imbalance); default: 1.0 3.0 2.0 3.0 2.0')

    # save path
    parser.add_argument('--save_path', type=str,
                        default='./cpts_multiclass_5class_v3/',
                        help='save path')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='pretrained weight path (single-class model)')

    return parser.parse_args()


def train_epoch(train_loader, model, criterion, optimizer, device, num_classes, cpc_weight=0.1, use_ds=True, use_cccpc=False):
    """Train one epoch (with deep supervision + CPC contrastive loss)"""
    model.train()
    total_loss = 0
    total_ce = 0
    total_tversky = 0
    total_focal = 0
    total_cpc = 0

    metrics_accum = MetricsAccumulator(num_classes)

    for i, (images, gts, edges) in enumerate(train_loader):
        images = Variable(images).to(device, non_blocking=True)
        gts = Variable(gts).to(device, non_blocking=True)
        edges_input = Variable(edges).to(device, non_blocking=True) if use_cccpc else None

        # compute the GT class labels (needed by the CPC module)
        gt_labels = torch.argmax(gts, dim=1)  # (B, H, W)

        optimizer.zero_grad()

        if use_cccpc:
            model_outputs = model(images, gt_labels=gt_labels, edge_map=edges_input)
        else:
            model_outputs = model(images, gt_labels=gt_labels)
        # model_outputs = (F1, F2, F3, F4, F5, edge, cpc_loss), all multi-class + the CPC loss
        outputs = model_outputs[0]  # F1 main output
        cpc_loss = model_outputs[-1] if isinstance(model_outputs, tuple) and len(model_outputs) >= 7 else None

        loss, ce, tversky, focal = criterion(outputs, gts)

        # Deep supervision: auxiliary losses for F2-F5 (with decreasing weights)
        if use_ds and isinstance(model_outputs, tuple) and len(model_outputs) >= 5:
            ds_weights = [0.4, 0.3, 0.2, 0.1]  # F2, F3, F4, F5
            for idx, ds_w in enumerate(ds_weights):
                aux_out = model_outputs[idx + 1]
                aux_loss, _, _, _ = criterion(aux_out, gts)
                loss = loss + ds_w * aux_loss

        # CPC contrastive loss (compatible with both scalar CPC and the FS-CPC dict format)
        if cpc_loss is not None:
            if isinstance(cpc_loss, dict):
                # FS-CPC: {'hf': ..., 'lf': ..., 'total': ...}
                loss = loss + cpc_weight * cpc_loss['total']
                total_cpc += cpc_loss['total'].item()
            else:
                # standard CPC: a single scalar
                loss = loss + cpc_weight * cpc_loss
                total_cpc += cpc_loss.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_tversky += tversky.item()
        total_focal += focal.item()

        # accumulate IoU with the accumulator (much faster than per-batch calculate_metrics)
        with torch.no_grad():
            metrics_accum.update(outputs, gts)

        if (i + 1) % 100 == 0:
            if cpc_loss is not None:
                cpc_str = f', CPC: {cpc_loss["total"].item():.4f}' if isinstance(cpc_loss, dict) else f', CPC: {cpc_loss.item():.4f}'
            else:
                cpc_str = ''
            print(f'    Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}{cpc_str}', flush=True)

        del images, gts, gt_labels, outputs, model_outputs, loss, ce, tversky, focal

    n = len(train_loader)
    train_metrics = metrics_accum.compute_metrics()
    return {
        'loss': total_loss / n,
        'ce': total_ce / n,
        'tversky': total_tversky / n,
        'focal': total_focal / n,
        'cpc': total_cpc / n,
        'iou': train_metrics['mean_iou']
    }


def validate(val_loader, model, criterion, device, num_classes):
    """Validate (with GPU memory management)"""
    model.eval()
    total_loss = 0
    total_ce = 0
    total_tversky = 0
    total_focal = 0

    metrics_accumulator = MetricsAccumulator(num_classes)

    with torch.no_grad():
        for images, gts, edges in val_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            model_outputs = model(images)
            if isinstance(model_outputs, tuple):
                outputs = model_outputs[0]
            else:
                outputs = model_outputs

            loss, ce, tversky, focal = criterion(outputs, gts)

            total_loss += loss.item()
            total_ce += ce.item()
            total_tversky += tversky.item()
            total_focal += focal.item()

            metrics_accumulator.update(outputs, gts)

            # release intermediate variables to avoid GPU memory buildup
            del images, gts, outputs, model_outputs, loss, ce, tversky, focal

    metrics = metrics_accumulator.compute_metrics()

    n = len(val_loader)
    return {
        'loss': total_loss / n,
        'ce': total_ce / n,
        'tversky': total_tversky / n,
        'focal': total_focal / n,
        'metrics': metrics
    }


def _provided_flags():
    """Return the set of flag names explicitly provided on the command line (so CLI overrides yaml)."""
    return {t.split('=')[0].lstrip('-') for t in sys.argv[1:] if t.startswith('-')}


def _apply_data_root(args):
    """Auto-fill the train/val Imgs/GT/Edge paths and save_path from data_root."""
    root = getattr(args, 'data_root', None)
    provided = _provided_flags()
    if root:
        root = root.rstrip('/')
        for arg_name, suffix in [
            ('rgb_root',     'train/Imgs'), ('gt_root',   'train/GT'),   ('edge_root',   'train/Edge'),
            ('val_rgb_root', 'val/Imgs'),   ('val_gt_root','val/GT'),    ('val_edge_root','val/Edge'),
        ]:
            if arg_name not in provided:
                setattr(args, arg_name, f'{root}/{suffix}/')

    # different configs save to different directories by default to avoid overwriting each other
    if args.config and 'save_path' not in provided:
        base = os.path.splitext(os.path.basename(args.config))[0]
        args.save_path = f'./cpts_{base}/'
    return args


def apply_config(args):
    """Read the YAML given by --config and map its values onto the argparse parameters.
    Rule: explicit CLI flags take precedence > YAML > built-in defaults."""
    if not args.config:
        return _apply_data_root(args)

    try:
        import yaml
    except ImportError:
        raise ImportError('--config requires PyYAML: pip install pyyaml')

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    provided = _provided_flags()

    def set_arg(name, value):
        """Override only when the value is non-empty and not explicitly provided on the CLI."""
        if value is None or name in provided:
            return
        setattr(args, name, value)

    data   = cfg.get('data',   {}) or {}
    train  = cfg.get('train',  {}) or {}
    cpc    = cfg.get('cpc',    {}) or {}
    fscpc  = cfg.get('fscpc',  {}) or {}
    cacpc  = cfg.get('cacpc',  {}) or {}
    model  = cfg.get('model',  {}) or {}

    # ---- model selection ----
    mname = model.get('name')
    if mname == 'NetFS_CPC':
        args.use_fscpc = True
    elif mname == 'NetCC_CPC':
        args.use_cccpc = True

    # ---- data root ----
    if isinstance(data.get('root'), str):
        args.data_root = data['root']

    # ---- scalar mapping (YAML key -> argparse name) ----
    set_arg('num_classes', model.get('num_classes') or data.get('num_classes'))
    set_arg('epoch',        train.get('epochs'))
    set_arg('batch_size',   train.get('batch_size'))
    set_arg('lr',           train.get('lr'))
    set_arg('trainsize',    data.get('train_size'))
    set_arg('patience',     (train.get('early_stop') or {}).get('patience'))

    # temperature: cpc.temperature -> cacpc.base_temperature -> fscpc.hf_temperature
    temp = cpc.get('temperature') or cacpc.get('base_temperature') or fscpc.get('hf_temperature')
    set_arg('temperature', temp)
    set_arg('cpc_weight', cpc.get('loss_weight') or fscpc.get('loss_weight') or cacpc.get('loss_weight'))
    set_arg('enable_refinement', cpc.get('use_refinement'))
    set_arg('hf_temperature', fscpc.get('hf_temperature'))
    set_arg('lf_temperature', fscpc.get('lf_temperature'))
    set_arg('load_pretrained', model.get('init_encoder'))

    return _apply_data_root(args)


def main():
    args = parse_args()
    args = apply_config(args)

    # class names
    class_names = {
        5: ['background', 'camouflage_soldier', 'military_vehicle', 'tank', 'fortification'],
        4: ['camouflage_soldier', 'military_vehicle', 'tank', 'fortification']
    }.get(args.num_classes, [f'Class_{i}' for i in range(args.num_classes)])

    print("=" * 70)
    print("SFCNet multi-class segmentation training (improved)")
    print("=" * 70)
    print(f"Number of classes: {args.num_classes}")
    print(f"Classes: {', '.join([f'{i}={name}' for i, name in enumerate(class_names)])}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"GPU: {args.gpu_id}")
    print(f"Early-stopping patience: {args.patience}")
    print(f"Loss weights: CE={args.ce_weight}, Tversky={args.tversky_weight}, Focal={args.focal_weight}, CPC={args.cpc_weight}")
    print(f"CPC params: temperature={args.temperature}, enable_refinement={args.enable_refinement}")
    print("=" * 70)

    # set up the device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True  # GPU optimization
    print(f"\nUsing device: {device}")
    print("cudnn.benchmark enabled for GPU performance")

    # create the save directory
    os.makedirs(args.save_path, exist_ok=True)

    # save the configuration
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # create the model
    print("\nBuilding model...")
    if args.use_fscpc:
        print(f"  using FS-CPC (frequency-sensitive prototype contrast)")
        print(f"  CPC params: temperature={args.temperature}, enable_refinement={args.enable_refinement}, cpc_weight={args.cpc_weight}")
        model = NetFS_CPC(
            num_classes=args.num_classes,
            temperature=args.temperature,
            enable_refinement=args.enable_refinement,
            hf_temperature=args.hf_temperature,
            lf_temperature=args.lf_temperature
        )
    elif args.use_cccpc:
        print(f"  using CC-CPC (confusion-aware boundary-enhanced prototype contrast)")
        print(f"  CPC params: temperature={args.temperature}, enable_refinement={args.enable_refinement}, cpc_weight={args.cpc_weight}")
        model = NetCC_CPC(
            num_classes=args.num_classes,
            temperature=args.temperature,
            enable_refinement=args.enable_refinement
        )
    else:
        print(f"  using standard CPC")
        print(f"  CPC params: temperature={args.temperature}, enable_refinement={args.enable_refinement}, cpc_weight={args.cpc_weight}")
        model = NetMultiClass(
            num_classes=args.num_classes,
            temperature=args.temperature,
            enable_refinement=args.enable_refinement
        )

    # load pretrained weights
    if args.load_pretrained:
        print(f"\nLoading pretrained weights: {args.load_pretrained}")
        if args.use_fscpc and hasattr(model, 'load_from_multiclass_weights'):
            model.load_from_multiclass_weights(args.load_pretrained)
        elif args.use_cccpc and hasattr(model, 'load_from_multiclass_weights'):
            model.load_from_multiclass_weights(args.load_pretrained)
        elif 'multi' in args.load_pretrained.lower():
            # multi-class weight loading (strict=False allows missing CPC params)
            state_dict = torch.load(args.load_pretrained, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f'Weights loaded (strict=False, CPC module randomly initialized)')
        else:
            model.load_from_single_class_weights(args.load_pretrained)

    model = model.to(device)

    # compute the parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # data loader
    print("\nLoading data...")
    # Use num_workers=0 on Windows to avoid multiprocessing issues and keep data loading smooth
    # enable augmentation for the training set, disable for the validation set
    train_loader = get_loader_multiclass(
        args.rgb_root, args.gt_root, args.edge_root,
        batchsize=args.batch_size, trainsize=args.trainsize,
        num_classes=args.num_classes, shuffle=True, num_workers=0,
        augmentation=True
    )

    val_loader = get_loader_multiclass(
        args.val_rgb_root, args.val_gt_root, args.val_edge_root,
        batchsize=args.batch_size, trainsize=args.trainsize,
        num_classes=args.num_classes, shuffle=False, num_workers=0,
        augmentation=False
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # loss function
    print("\nSetting up loss function...")
    if args.class_weights:
        print(f"Using class weights: {args.class_weights}")
    criterion = CombinedLoss(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        tversky_weight=args.tversky_weight,
        focal_weight=args.focal_weight,
        class_weights=args.class_weights
    )

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    # resume: check whether a checkpoint exists
    start_epoch = 0
    best_val_iou = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'lr': [],
        'per_class_iou': []
    }

    checkpoint_path = os.path.join(args.save_path, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint file: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        start_epoch = ckpt['epoch']
        best_val_iou = ckpt.get('best_val_iou', 0)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        history = ckpt.get('history', history)
        print(f"Resuming from epoch {start_epoch}, best historical IoU: {best_val_iou:.4f}")

    # early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    start_time = datetime.now()

    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(start_epoch, args.epoch):
        epoch_start = datetime.now()

        # clear the GPU cache before each epoch to avoid fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        max_retries = 3
        for retry in range(max_retries):
            try:
                # training
                train_stats = train_epoch(train_loader, model, criterion, optimizer, device, args.num_classes, cpc_weight=args.cpc_weight, use_cccpc=args.use_cccpc)

                # clear the cache after training before validating
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # validation
                val_stats = validate(val_loader, model, criterion, device, args.num_classes)
                break
            except RuntimeError as e:
                if 'CUDA' in str(e) and retry < max_retries - 1:
                    print(f"\n  CUDA error (retry {retry+1}/{max_retries}): {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import time
                    time.sleep(10)
                    continue
                else:
                    # if the last retry also fails, save the current state
                    print(f"\n  Unrecoverable CUDA error, saving checkpoint...")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_iou': best_val_iou,
                        'history': history
                    }, checkpoint_path)
                    raise

        # update the learning rate
        scheduler.step(val_stats['loss'])
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # record the history
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['train_iou'].append(train_stats['iou'])
        history['val_iou'].append(val_stats['metrics']['mean_iou'])
        history['lr'].append(current_lr)
        history['per_class_iou'].append(val_stats['metrics']['per_class_iou'])

        # print the results
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\nEpoch [{epoch+1:03d}/{args.epoch}] ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_stats['loss']:.4f}, IoU: {train_stats['iou']:.4f}")
        print(f"  Val   - Loss: {val_stats['loss']:.4f}, IoU: {val_stats['metrics']['mean_iou']:.4f}")
        print(f"  Losses - CE: {val_stats['ce']:.4f}, Tversky: {val_stats['tversky']:.4f}, Focal: {val_stats['focal']:.4f}, CPC: {train_stats['cpc']:.4f}")
        print(f"  Per-class IoU:")
        for i, iou in enumerate(val_stats['metrics']['per_class_iou']):
            print(f"    Class {i} ({class_names[i]}): {iou:.4f}")
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory - used: {mem_alloc:.2f}GB, allocated: {mem_reserved:.2f}GB")
        print(f"  LR: {current_lr:.2e}")

        # save the best model
        if val_stats['metrics']['mean_iou'] > best_val_iou:
            best_val_iou = val_stats['metrics']['mean_iou']
            torch.save(model.state_dict(), os.path.join(args.save_path, 'Net_multi_best_iou.pth'))
            print(f"  -> saved best-IoU model: {best_val_iou:.4f}")

        # save a checkpoint every 5 epochs (for resuming)
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'history': history
            }, checkpoint_path)

        # periodic save
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'Net_multi_epoch_{epoch+1}.pth'))

        # early-stopping check (monitor val IoU)
        early_stopping(val_stats['metrics']['mean_iou'], model, args.save_path)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered, stopping training!")
            break

    # training complete
    total_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Total training time: {total_time}")
    print(f"Best val IoU: {best_val_iou:.4f}")
    print(f"Best val IoU (early stopping): {early_stopping.best_score:.4f}")
    print("=" * 70)

    # save the training history
    with open(os.path.join(args.save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining history saved to: {args.save_path}/training_history.json")


if __name__ == '__main__':
    main()
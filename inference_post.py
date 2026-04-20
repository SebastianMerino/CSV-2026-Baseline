#!/usr/bin/env python3
"""
inference_post.py

Two-stage inference and evaluation:
  1. Frozen segmentation model produces seg masks
  2. Post-segmentation classifier predicts vulnerability

Usage:
    python inference_post.py \
        --eval_json ./data/test.json \
        --seg_checkpoint ./checkpoints/best.pth \
        --cls_checkpoint ./checkpoints_post/best_opt.pth \
        --save_preds --save_grid --gpu 0
"""

import os
import json
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from scipy.ndimage import distance_transform_edt, binary_erosion

from model.unet import UNetTwoView
from model.unet_v2 import UNetTwoViewV2
from model.post_seg_classifier import get_post_classifier

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_image(img):
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def compute_dice(pred, gt, cls):
    p = (pred == cls)
    g = (gt == cls)
    inter = (p & g).sum()
    union = p.sum() + g.sum()
    return 2.0 * inter / (union + 1e-8)


def compute_nsd(pred_bin, gt_bin, tolerance=3.0):
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0
    
    border_pred = pred_bin ^ binary_erosion(pred_bin)
    border_gt = gt_bin ^ binary_erosion(gt_bin)
    
    if border_pred.sum() == 0 or border_gt.sum() == 0:
        return 0.0
    
    dt_gt = distance_transform_edt(~border_gt)
    dt_pred = distance_transform_edt(~border_pred)
    
    pred_to_gt = (dt_gt[border_pred] <= tolerance).mean()
    gt_to_pred = (dt_pred[border_gt] <= tolerance).mean()
    
    return (pred_to_gt + gt_to_pred) / 2.0


SEG_PALETTE = {0: None, 1: (1.0, 0.2, 0.2), 2: (0.2, 0.9, 0.2)}
SEG_ALPHA = 0.45


def overlay_mask(base_grey, mask):
    rgb = np.stack([base_grey] * 3, axis=-1)
    for cls_idx, color in SEG_PALETTE.items():
        if color is None:
            continue
        region = mask == cls_idx
        if region.any():
            for c in range(3):
                rgb[..., c][region] = rgb[..., c][region] * (1 - SEG_ALPHA) + color[c] * SEG_ALPHA
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (matching CSVSemiDataset exactly)
# ─────────────────────────────────────────────────────────────────────────────

def read_pair(image_h5_file):
    """Read image pair from h5 file - matches CSVSemiDataset._read_pair"""
    with h5py.File(image_h5_file, 'r') as f:
        long_img = f['long_img'][:]
        trans_img = f['trans_img'][:]
    
    long_img = long_img.astype(np.float32)
    trans_img = trans_img.astype(np.float32)
    
    if long_img.max() > 1.0:
        long_img = long_img / 255.0
    if trans_img.max() > 1.0:
        trans_img = trans_img / 255.0
    
    return long_img, trans_img


def read_label(label_h5_file):
    """Read labels from h5 file - matches CSVSemiDataset._read_label"""
    with h5py.File(label_h5_file, 'r') as f:
        long_mask = f['long_mask'][:]
        trans_mask = f['trans_mask'][:]
        cls = f['cls'][()]
    
    # Map mask values {0, 128, 255} -> {0, 1, 2}
    long_mask = long_mask.astype(np.int64)
    trans_mask = trans_mask.astype(np.int64)
    
    long_mask = np.where(long_mask == 128, 1, long_mask)
    long_mask = np.where(long_mask == 255, 2, long_mask)
    trans_mask = np.where(trans_mask == 128, 1, trans_mask)
    trans_mask = np.where(trans_mask == 255, 2, trans_mask)
    
    # Handle cls shape
    cls = int(np.array(cls).flat[0])
    
    return long_mask, trans_mask, cls


class EvalJSONDataset:
    """Dataset for evaluation using JSON file with ground truth."""
    
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.case_list = json.load(f)
        if len(self.case_list) == 0:
            raise ValueError(f"No entries found in {json_path}")
        self.has_gt = True

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        image_path = case['image']
        label_path = case.get('label', None)
        
        # Read images
        long_img, trans_img = read_pair(image_path)
        
        # Read labels if available
        long_mask_gt, trans_mask_gt, cls_gt = None, None, None
        if label_path and os.path.exists(label_path):
            long_mask_gt, trans_mask_gt, cls_gt = read_label(label_path)
        
        # Convert to tensors
        long_t = torch.from_numpy(long_img).unsqueeze(0).float()
        trans_t = torch.from_numpy(trans_img).unsqueeze(0).float()
        
        return {
            'path': image_path,
            'filename': os.path.basename(image_path),
            'long_t': long_t,
            'trans_t': trans_t,
            'long_shape': long_img.shape,
            'trans_shape': trans_img.shape,
            'cls_gt': cls_gt,
            'long_mask_gt': long_mask_gt,
            'trans_mask_gt': trans_mask_gt,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_seg_model(args):
    if args.model == "UNet":
        return UNetTwoView(in_chns=1, seg_class_num=args.seg_num_classes, cls_class_num=args.cls_num_classes)
    elif args.model == "UNetV2":
        return UNetTwoViewV2(in_chns=1, seg_class_num=args.seg_num_classes, cls_class_num=args.cls_num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def load_checkpoint(model, ckpt_path, device, strict=False):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt.get("classifier", ckpt)))
    else:
        state = ckpt
    
    model.load_state_dict(state, strict=strict)
    return model, ckpt


def load_seg_model(args, device):
    print(f"Loading segmentation model: {args.model} from {args.seg_checkpoint}")
    model = get_seg_model(args)
    model, _ = load_checkpoint(model, args.seg_checkpoint, device, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_cls_model(args, device):
    print(f"Loading classifier: {args.classifier} from {args.cls_checkpoint}")
    classifier = get_post_classifier(
        variant=args.classifier,
        seg_classes=args.seg_num_classes,
        cls_classes=args.cls_num_classes,
        dropout=0.0,
    )
    classifier, ckpt = load_checkpoint(classifier, args.cls_checkpoint, device, strict=True)
    
    if "best_thresh" in ckpt and args.cls_threshold is None:
        args.cls_threshold = ckpt["best_thresh"]
        print(f"  Using saved threshold: {args.cls_threshold:.3f}")
    
    return classifier.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(seg_model, classifier, device, sample, resize_target, cls_threshold, use_tta=False):
    long_t = sample['long_t'].unsqueeze(0).to(device)
    trans_t = sample['trans_t'].unsqueeze(0).to(device)
    long_shape = sample['long_shape']
    trans_shape = sample['trans_shape']
    
    xL_r = F.interpolate(long_t, (resize_target, resize_target), mode="bilinear", align_corners=False)
    xT_r = F.interpolate(trans_t, (resize_target, resize_target), mode="bilinear", align_corners=False)
    
    segL_logits, segT_logits, _ = seg_model(xL_r, xT_r)
    
    if use_tta:
        segL_f, segT_f, _ = seg_model(xL_r.flip(-1), xT_r.flip(-1))
        segL_logits = (segL_logits + segL_f.flip(-1)) / 2
        segT_logits = (segT_logits + segT_f.flip(-1)) / 2
    
    cls_out = classifier(xL_r, xT_r, segL_logits, segT_logits)
    cls_prob = torch.sigmoid(cls_out).cpu().item()
    cls_pred = 1 if cls_prob >= cls_threshold else 0
    
    segL_up = F.interpolate(segL_logits, long_shape, mode="bilinear", align_corners=False)
    segT_up = F.interpolate(segT_logits, trans_shape, mode="bilinear", align_corners=False)
    
    pred_long = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_trans = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    return pred_long, pred_trans, cls_pred, cls_prob


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results, seg_classes=3):
    metrics = {}
    
    # Classification
    cls_gts = [r['cls_gt'] for r in results if r['cls_gt'] is not None]
    cls_preds = [r['cls_pred'] for r in results if r['cls_gt'] is not None]
    
    if cls_gts:
        metrics['cls_accuracy'] = sum(g == p for g, p in zip(cls_gts, cls_preds)) / len(cls_gts)
        metrics['cls_f1'] = f1_score(cls_gts, cls_preds, zero_division=0)
        
        cls_probs = [r['cls_prob'] for r in results if r['cls_gt'] is not None]
        if len(set(cls_gts)) > 1:
            metrics['cls_auc'] = roc_auc_score(cls_gts, cls_probs)
        
        metrics['confusion_matrix'] = confusion_matrix(cls_gts, cls_preds)
    
    # Segmentation
    seg_dice = defaultdict(list)
    seg_nsd = defaultdict(list)
    
    for r in results:
        for view, pred_key, gt_key in [('long', 'pred_long', 'long_mask_gt'), 
                                        ('trans', 'pred_trans', 'trans_mask_gt')]:
            gt = r[gt_key]
            if gt is not None:
                pred = r[pred_key]
                for c in range(1, seg_classes):
                    seg_dice[f'dice_{view}_c{c}'].append(compute_dice(pred, gt, c))
                    nsd = compute_nsd((pred == c), (gt == c), tolerance=3.0)
                    seg_nsd[f'nsd_{view}_c{c}'].append(nsd)
    
    for key, values in {**seg_dice, **seg_nsd}.items():
        if values:
            metrics[key] = np.mean(values)
    
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_qualitative_grid(results, out_path, num_samples=8):
    has_gt = any(r['long_mask_gt'] is not None for r in results)
    selected = results[:min(len(results), num_samples)]
    
    n_cols = len(selected)
    n_rows = 6 if has_gt else 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for col, r in enumerate(selected):
        long_img, trans_img = read_pair(r['path'])
        long_img = normalize_image(long_img)
        trans_img = normalize_image(trans_img)
        
        row = 0
        
        # Long input
        axes[row, col].imshow(long_img, cmap='gray')
        axes[row, col].set_title("Long Input", fontsize=8)
        axes[row, col].axis('off')
        row += 1
        
        # Long pred
        axes[row, col].imshow(overlay_mask(long_img, r['pred_long']))
        axes[row, col].set_title("Long Pred", fontsize=8)
        axes[row, col].axis('off')
        row += 1
        
        # Long GT
        if has_gt:
            if r['long_mask_gt'] is not None:
                axes[row, col].imshow(overlay_mask(long_img, r['long_mask_gt']))
            else:
                axes[row, col].imshow(long_img, cmap='gray')
            axes[row, col].set_title("Long GT", fontsize=8)
            axes[row, col].axis('off')
            row += 1
        
        # Trans input
        axes[row, col].imshow(trans_img, cmap='gray')
        axes[row, col].set_title("Trans Input", fontsize=8)
        axes[row, col].axis('off')
        row += 1
        
        # Trans pred
        axes[row, col].imshow(overlay_mask(trans_img, r['pred_trans']))
        axes[row, col].set_title("Trans Pred", fontsize=8)
        axes[row, col].axis('off')
        row += 1
        
        # Trans GT
        if has_gt:
            if r['trans_mask_gt'] is not None:
                axes[row, col].imshow(overlay_mask(trans_img, r['trans_mask_gt']))
            else:
                axes[row, col].imshow(trans_img, cmap='gray')
            axes[row, col].set_title("Trans GT", fontsize=8)
            axes[row, col].axis('off')
        
        # Classification info in title
        cls_str = f"Pred: {'Vuln' if r['cls_pred'] else 'Stable'} ({r['cls_prob']:.2f})"
        if r['cls_gt'] is not None:
            gt_str = 'Vuln' if r['cls_gt'] else 'Stable'
            match = '✓' if r['cls_pred'] == r['cls_gt'] else '✗'
            cls_str = f"GT: {gt_str} | {cls_str} {match}"
        axes[0, col].set_title(cls_str, fontsize=7)
    
    fig.legend(handles=[mpatches.Patch(color=SEG_PALETTE[1], label="Plaque"),
                        mpatches.Patch(color=SEG_PALETTE[2], label="Vessel")],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Two-stage inference: Segmentation + Classifier")
    
    # Data
    parser.add_argument("--eval_json", type=str, required=True, help="JSON file with image/label paths")
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Models
    parser.add_argument("--seg_checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="UNetV2", choices=["UNet", "UNetV2"])
    parser.add_argument("--cls_checkpoint", type=str, required=True)
    parser.add_argument("--classifier", type=str, default="full", choices=["full", "seg_only", "seg_img", "light"])
    parser.add_argument("--cls_threshold", type=float, default=None)
    
    # Settings
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize_target", type=int, default=256)
    
    # Options
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--save_grid", action="store_true")
    parser.add_argument("--num_grid_samples", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset
    print(f"Loading from JSON: {args.eval_json}")
    dataset = EvalJSONDataset(args.eval_json)
    
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.eval_json), "preds_test")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir} | Samples: {len(dataset)}")
    
    # Models
    seg_model = load_seg_model(args, device)
    classifier = load_cls_model(args, device)
    
    if args.cls_threshold is None:
        args.cls_threshold = 0.5
    print(f"Threshold: {args.cls_threshold:.3f} | TTA: {args.use_tta}")
    
    # Inference
    print("\nRunning inference...")
    results = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        pred_long, pred_trans, cls_pred, cls_prob = predict_single(
            seg_model, classifier, device, sample,
            args.resize_target, args.cls_threshold, args.use_tta
        )
        
        results.append({
            'filename': sample['filename'],
            'path': sample['path'],
            'pred_long': pred_long,
            'pred_trans': pred_trans,
            'cls_pred': cls_pred,
            'cls_prob': cls_prob,
            'cls_gt': sample['cls_gt'],
            'long_mask_gt': sample['long_mask_gt'],
            'trans_mask_gt': sample['trans_mask_gt'],
        })
        
        # Progress
        gt_info = f" | GT cls={sample['cls_gt']}" if sample['cls_gt'] is not None else ""
        print(f"[{idx+1:3d}/{len(dataset)}] {sample['filename']} | pred={cls_pred} prob={cls_prob:.3f}{gt_info}")
    
    # Metrics
    metrics = compute_metrics(results, args.seg_num_classes)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if 'cls_accuracy' in metrics:
        print(f"Classification Accuracy: {metrics['cls_accuracy']:.4f}")
        print(f"Classification F1:       {metrics['cls_f1']:.4f}")
        if 'cls_auc' in metrics:
            print(f"Classification AUC:      {metrics['cls_auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    print("\nSegmentation Dice:")
    for key in sorted(k for k in metrics if k.startswith('dice_')):
        print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nSegmentation NSD:")
    for key in sorted(k for k in metrics if k.startswith('nsd_')):
        print(f"  {key}: {metrics[key]:.4f}")
    
    # Save outputs
    if args.save_preds:
        preds_path = os.path.join(out_dir, "predictions.json")
        save_data = [{
            'filename': r['filename'],
            'cls_pred': r['cls_pred'],
            'cls_prob': float(r['cls_prob']),
            'cls_gt': r['cls_gt'],
        } for r in results]
        with open(preds_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved predictions: {preds_path}")
    
    if args.save_grid:
        grid_path = os.path.join(out_dir, "qualitative_grid.png")
        save_qualitative_grid(results, grid_path, args.num_grid_samples)
    
    # Save metrics
    metrics_save = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
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
# NEW: Helper function to convert mask to logits (one-hot style)
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_logits(mask, num_classes, device, resize_target):
    """
    Convert a ground truth segmentation mask to "pseudo-logits" that can be
    fed into the classifier in place of predicted segmentation logits.
    
    Args:
        mask: (H, W) numpy array with integer class labels
        num_classes: number of segmentation classes
        device: torch device
        resize_target: target size for resizing
    
    Returns:
        logits: (1, num_classes, resize_target, resize_target) tensor
                with high values (e.g., 10) for the correct class and low (-10) for others
    """
    H, W = mask.shape
    
    # Create one-hot encoding: (H, W) -> (num_classes, H, W)
    one_hot = np.zeros((num_classes, H, W), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (mask == c).astype(np.float32)
    
    # Convert to tensor and add batch dimension: (1, num_classes, H, W)
    logits = torch.from_numpy(one_hot).unsqueeze(0).to(device)
    
    # Resize to match expected input size
    logits = F.interpolate(logits, (resize_target, resize_target), mode="bilinear", align_corners=False)
    
    # Scale to logit-like values (high confidence)
    # Using large values so softmax gives ~1.0 for correct class
    logits = logits * 20 - 10  # Maps 0->-10, 1->10
    
    return logits

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MODIFIED: predict_single with option to use GT segmentation for classification
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(seg_model, classifier, device, sample, resize_target, cls_threshold, 
                   use_tta=False, seg_num_classes=3):
    """
    Run inference on a single sample.
    
    Returns predictions using:
    1. Predicted segmentation (normal operation)
    2. Ground truth segmentation (oracle mode, if GT available)
    """
    long_t = sample['long_t'].unsqueeze(0).to(device)
    trans_t = sample['trans_t'].unsqueeze(0).to(device)
    long_shape = sample['long_shape']
    trans_shape = sample['trans_shape']
    
    # Resize images
    xL_r = F.interpolate(long_t, (resize_target, resize_target), mode="bilinear", align_corners=False)
    xT_r = F.interpolate(trans_t, (resize_target, resize_target), mode="bilinear", align_corners=False)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Normal prediction: Use model's segmentation
    # ─────────────────────────────────────────────────────────────────────────
    segL_logits, segT_logits, _ = seg_model(xL_r, xT_r)
    
    if use_tta:
        segL_f, segT_f, _ = seg_model(xL_r.flip(-1), xT_r.flip(-1))
        segL_logits = (segL_logits + segL_f.flip(-1)) / 2
        segT_logits = (segT_logits + segT_f.flip(-1)) / 2
    
    # Classification with PREDICTED segmentation
    cls_out = classifier(xL_r, xT_r, segL_logits, segT_logits)
    cls_prob = torch.sigmoid(cls_out).cpu().item()
    cls_pred = 1 if cls_prob >= cls_threshold else 0
    
    # Get segmentation predictions
    segL_up = F.interpolate(segL_logits, long_shape, mode="bilinear", align_corners=False)
    segT_up = F.interpolate(segT_logits, trans_shape, mode="bilinear", align_corners=False)
    pred_long = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_trans = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. Oracle prediction: Use GT segmentation (if available)
    # ─────────────────────────────────────────────────────────────────────────
    cls_pred_oracle = None
    cls_prob_oracle = None
    
    if sample['long_mask_gt'] is not None and sample['trans_mask_gt'] is not None:
        # Convert GT masks to pseudo-logits
        gt_segL_logits = mask_to_logits(sample['long_mask_gt'], seg_num_classes, device, resize_target)
        gt_segT_logits = mask_to_logits(sample['trans_mask_gt'], seg_num_classes, device, resize_target)
        
        # Classification with GROUND TRUTH segmentation
        cls_out_oracle = classifier(xL_r, xT_r, gt_segL_logits, gt_segT_logits)
        cls_prob_oracle = torch.sigmoid(cls_out_oracle).cpu().item()
        cls_pred_oracle = 1 if cls_prob_oracle >= cls_threshold else 0
    
    return {
        # Segmentation predictions
        'pred_long': pred_long,
        'pred_trans': pred_trans,
        # Classification with predicted segmentation
        'cls_pred': cls_pred,
        'cls_prob': cls_prob,
        # Classification with GT segmentation (oracle)
        'cls_pred_oracle': cls_pred_oracle,
        'cls_prob_oracle': cls_prob_oracle,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results, seg_classes=3):
    """
    MODIFIED FUNCTION TO ALSO COMPUTE OFFICIAL CSV 2026 CHALLENGE SCORES.
    
    Segmentation Score (S_seg):
        S_v,vessel = (DSC_vessel + NSD_vessel) / 2
        S_v,plaque = (DSC_plaque + NSD_plaque) / 2  
        S_v = 0.4 * S_v,vessel + 0.6 * S_v,plaque
        S_seg = (S_long + S_trans) / 2
    
    Classification Score (S_cls):
        Average F1-score across all classes
    """
    metrics = {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Classification with PREDICTED segmentation
    # ─────────────────────────────────────────────────────────────────────────
    cls_gts = [r['cls_gt'] for r in results if r['cls_gt'] is not None]
    cls_preds = [r['cls_pred'] for r in results if r['cls_gt'] is not None]
    cls_probs = [r['cls_prob'] for r in results if r['cls_gt'] is not None]
    
    if cls_gts:
        metrics['cls_accuracy'] = sum(g == p for g, p in zip(cls_gts, cls_preds)) / len(cls_gts)
        metrics['cls_f1'] = f1_score(cls_gts, cls_preds, zero_division=0)
        
        if len(set(cls_gts)) > 1:
            metrics['cls_auc'] = roc_auc_score(cls_gts, cls_probs)
        
        metrics['confusion_matrix'] = confusion_matrix(cls_gts, cls_preds)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Classification with GT segmentation (ORACLE)
    # ─────────────────────────────────────────────────────────────────────────
    cls_preds_oracle = [r['cls_pred_oracle'] for r in results 
                        if r['cls_gt'] is not None and r['cls_pred_oracle'] is not None]
    cls_probs_oracle = [r['cls_prob_oracle'] for r in results 
                        if r['cls_gt'] is not None and r['cls_prob_oracle'] is not None]
    cls_gts_oracle = [r['cls_gt'] for r in results 
                      if r['cls_gt'] is not None and r['cls_pred_oracle'] is not None]
    
    if cls_gts_oracle:
        metrics['cls_accuracy_oracle'] = sum(g == p for g, p in zip(cls_gts_oracle, cls_preds_oracle)) / len(cls_gts_oracle)
        metrics['cls_f1_oracle'] = f1_score(cls_gts_oracle, cls_preds_oracle, zero_division=0)
        
        if len(set(cls_gts_oracle)) > 1:
            metrics['cls_auc_oracle'] = roc_auc_score(cls_gts_oracle, cls_probs_oracle)
        
        metrics['confusion_matrix_oracle'] = confusion_matrix(cls_gts_oracle, cls_preds_oracle)
        
        # ─────────────────────────────────────────────────────────────────────
        # Performance gap analysis
        # ─────────────────────────────────────────────────────────────────────
        metrics['cls_accuracy_gap'] = metrics['cls_accuracy_oracle'] - metrics['cls_accuracy']
        metrics['cls_f1_gap'] = metrics['cls_f1_oracle'] - metrics['cls_f1']
        
        if 'cls_auc' in metrics and 'cls_auc_oracle' in metrics:
            metrics['cls_auc_gap'] = metrics['cls_auc_oracle'] - metrics['cls_auc']
    
    # ─────────────────────────────────────────────────────────────────────────
    # Segmentation metrics (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Per-sample analysis: Correlation between seg quality and cls correctness
    # ─────────────────────────────────────────────────────────────────────────
    per_sample_data = []
    for r in results:
        if r['cls_gt'] is None or r['long_mask_gt'] is None:
            continue
        
        # Calculate average segmentation score for this sample
        dice_scores = []
        nsd_scores = []
        for view, pred_key, gt_key in [('long', 'pred_long', 'long_mask_gt'), 
                                        ('trans', 'pred_trans', 'trans_mask_gt')]:
            gt = r[gt_key]
            pred = r[pred_key]
            for c in range(1, seg_classes):
                dice_scores.append(compute_dice(pred, gt, c))
                nsd_scores.append(compute_nsd((pred == c), (gt == c), tolerance=3.0))
        
        avg_dice = np.mean(dice_scores)
        avg_nsd = np.mean(nsd_scores)
        cls_correct = int(r['cls_pred'] == r['cls_gt'])
        cls_correct_oracle = int(r['cls_pred_oracle'] == r['cls_gt']) if r['cls_pred_oracle'] is not None else None
        
        per_sample_data.append({
            'avg_dice': avg_dice,
            'avg_nsd': avg_nsd,
            'cls_correct': cls_correct,
            'cls_correct_oracle': cls_correct_oracle,
            'cls_pred_changed': r['cls_pred'] != r['cls_pred_oracle'] if r['cls_pred_oracle'] is not None else None,
        })
    
    if per_sample_data:
        # Samples where prediction changed when using GT segmentation
        changed_samples = [s for s in per_sample_data if s['cls_pred_changed']]
        metrics['num_cls_predictions_changed'] = len(changed_samples)
        metrics['pct_cls_predictions_changed'] = len(changed_samples) / len(per_sample_data) * 100
        
        # Average segmentation quality for correct vs incorrect classifications
        correct_samples = [s for s in per_sample_data if s['cls_correct'] == 1]
        incorrect_samples = [s for s in per_sample_data if s['cls_correct'] == 0]
        
        if correct_samples:
            metrics['avg_dice_when_cls_correct'] = np.mean([s['avg_dice'] for s in correct_samples])
            metrics['avg_nsd_when_cls_correct'] = np.mean([s['avg_nsd'] for s in correct_samples])
        
        if incorrect_samples:
            metrics['avg_dice_when_cls_incorrect'] = np.mean([s['avg_dice'] for s in incorrect_samples])
            metrics['avg_nsd_when_cls_incorrect'] = np.mean([s['avg_nsd'] for s in incorrect_samples])
    
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
    
    # Option to run segmentation impact analysis
    parser.add_argument("--analyze_seg_impact", action="store_true",
                        help="Analyze impact of segmentation quality on classification")   

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

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
    
    if args.analyze_seg_impact:
        print("Segmentation impact analysis ENABLED")

    # Inference
    print("\nRunning inference...")
    results = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        preds = predict_single(
            seg_model, classifier, device, sample,
            args.resize_target, args.cls_threshold, args.use_tta,
            seg_num_classes=args.seg_num_classes  # Also missing this argument!
        )
        
        results.append({
            'filename': sample['filename'],
            'path': sample['path'],
            'pred_long': preds['pred_long'],
            'pred_trans': preds['pred_trans'],
            'cls_pred': preds['cls_pred'],
            'cls_prob': preds['cls_prob'],
            'cls_pred_oracle': preds['cls_pred_oracle'],
            'cls_prob_oracle': preds['cls_prob_oracle'],
            'cls_gt': sample['cls_gt'],
            'long_mask_gt': sample['long_mask_gt'],
            'trans_mask_gt': sample['trans_mask_gt'],
        })
        
        # Progress
        gt_info = f" | GT={sample['cls_gt']}" if sample['cls_gt'] is not None else ""
        oracle_info = ""
        if preds['cls_pred_oracle'] is not None:
            match = "=" if preds['cls_pred'] == preds['cls_pred_oracle'] else "≠"
            oracle_info = f" | oracle={preds['cls_pred_oracle']}({match})"
        print(f"[{idx+1:3d}/{len(dataset)}] {sample['filename']} | pred={preds['cls_pred']} prob={preds['cls_prob']:.3f}{oracle_info}{gt_info}")
    
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
    '''
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
    '''
    ### ADD MORE PER/FILE STATISTICS
    # Save outputs
    if args.save_preds:
        preds_path = os.path.join(out_dir, "predictions.json")
        save_data = []
        
        for r in results:
            sample_data = {
                'filename': r['filename'],
                # Classification with predicted segmentation
                'cls_pred': r['cls_pred'],
                'cls_prob': float(r['cls_prob']),
                # Classification with GT segmentation (oracle)
                'cls_pred_oracle': r['cls_pred_oracle'],
                'cls_prob_oracle': float(r['cls_prob_oracle']) if r['cls_prob_oracle'] is not None else None,
                # Ground truth
                'cls_gt': r['cls_gt'],
                # Analysis fields
                'pred_changed': r['cls_pred'] != r['cls_pred_oracle'] if r['cls_pred_oracle'] is not None else None,
                'prob_diff': float(r['cls_prob_oracle'] - r['cls_prob']) if r['cls_prob_oracle'] is not None else None,
                'cls_correct': r['cls_pred'] == r['cls_gt'] if r['cls_gt'] is not None else None,
                'cls_correct_oracle': r['cls_pred_oracle'] == r['cls_gt'] if (r['cls_gt'] is not None and r['cls_pred_oracle'] is not None) else None,
            }
            
            # Add per-sample segmentation metrics if GT available
            if r['long_mask_gt'] is not None and r['trans_mask_gt'] is not None:
                seg_metrics = {}
                for view, pred_key, gt_key in [('long', 'pred_long', 'long_mask_gt'), 
                                                ('trans', 'pred_trans', 'trans_mask_gt')]:
                    pred = r[pred_key]
                    gt = r[gt_key]
                    for c in range(1, args.seg_num_classes):
                        class_name = 'vessel' if c == 1 else 'plaque'
                        seg_metrics[f'dice_{view}_{class_name}'] = float(compute_dice(pred, gt, c))
                        seg_metrics[f'nsd_{view}_{class_name}'] = float(compute_nsd((pred == c), (gt == c), tolerance=3.0))
                
                # Average scores
                seg_metrics['avg_dice'] = float(np.mean([v for k, v in seg_metrics.items() if k.startswith('dice_')]))
                seg_metrics['avg_nsd'] = float(np.mean([v for k, v in seg_metrics.items() if k.startswith('nsd_')]))
                
                sample_data['segmentation'] = seg_metrics
            
            save_data.append(sample_data)
        
        with open(preds_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved predictions: {preds_path}")
    ###

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
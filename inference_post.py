#!/usr/bin/env python3
"""
inference_post.py

Two-stage inference:
  1. Frozen segmentation model produces seg masks
  2. Post-segmentation classifier predicts vulnerability

Usage:
    python inference_post.py \
        --val-dir ./data/val \
        --seg_checkpoint ./checkpoints/best_seg.pth \
        --cls_checkpoint ./checkpoints_post/full_xxx/best_opt.pth \
        --model UNetV2 \
        --classifier full \
        --gpu 0
"""

import os
import glob
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Local imports
from model.unet import UNetTwoView
from model.unet_v2 import UNetTwoViewV2
from model.Echocare import Echocare_UniMatch
from model.post_seg_classifier import get_post_classifier


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ValH5Dataset:
    """
    Simple dataset to iterate over validation image .h5 files.
    Each file contains 'long_img' and 'trans_img' numpy arrays.
    """
    
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.paths = sorted(glob.glob(os.path.join(images_dir, "*.h5")))
        
        if len(self.paths) == 0:
            raise ValueError(f"No .h5 files found in {images_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with h5py.File(p, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]
        
        # Original shapes
        long_shape = long_img.shape
        trans_shape = trans_img.shape
        
        # To torch tensors: [1, H, W], float32
        long_t = torch.from_numpy(long_img).unsqueeze(0).float()
        trans_t = torch.from_numpy(trans_img).unsqueeze(0).float()
        
        # Normalize to [0, 1] if values are in [0, 255]
        if long_t.max() > 1.0:
            long_t = long_t / 255.0
        if trans_t.max() > 1.0:
            trans_t = trans_t / 255.0
        
        return p, long_t, trans_t, long_shape, trans_shape


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_seg_model(args):
    """Create segmentation model architecture."""
    if args.model == "Echocare":
        model = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.encoder_pth,
        )
    elif args.model == "UNet":
        model = UNetTwoView(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    elif args.model == "UNetV2":
        model = UNetTwoViewV2(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model


def load_checkpoint(model, ckpt_path, device, strict=False):
    """Load model checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "classifier" in ckpt:
            state = ckpt["classifier"]
        else:
            state = ckpt
    else:
        state = ckpt
    
    model.load_state_dict(state, strict=strict)
    return model, ckpt


def load_seg_model(args, device):
    """Load and freeze segmentation model."""
    print(f"Loading segmentation model: {args.model}")
    print(f"  Checkpoint: {args.seg_checkpoint}")
    
    model = get_seg_model(args)
    model, ckpt = load_checkpoint(model, args.seg_checkpoint, device, strict=False)
    
    if "seg_score" in ckpt:
        print(f"  Seg score: {ckpt['seg_score']:.4f}")
    if "epoch" in ckpt:
        print(f"  Trained for {ckpt['epoch']} epochs")
    
    model = model.to(device)
    model.eval()
    
    # Freeze
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def load_cls_model(args, device):
    """Load post-segmentation classifier."""
    print(f"Loading classifier: {args.classifier}")
    print(f"  Checkpoint: {args.cls_checkpoint}")
    
    classifier = get_post_classifier(
        variant=args.classifier,
        seg_classes=args.seg_num_classes,
        cls_classes=args.cls_num_classes,
        dropout=0.0,  # No dropout during inference
    )
    
    classifier, ckpt = load_checkpoint(classifier, args.cls_checkpoint, device, strict=True)
    
    if "f1_best" in ckpt:
        print(f"  Best F1: {ckpt['f1_best']:.4f}")
    if "best_thresh" in ckpt:
        print(f"  Optimal threshold: {ckpt['best_thresh']:.2f}")
        args.cls_threshold = ckpt["best_thresh"]
    if "epoch" in ckpt:
        print(f"  Trained for {ckpt['epoch']} epochs")
    
    classifier = classifier.to(device)
    classifier.eval()
    
    return classifier


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def unpack_cls(cls_out):
    """Handle different classifier output formats."""
    if isinstance(cls_out, (tuple, list)):
        return cls_out[0]
    return cls_out


@torch.no_grad()
def predict_single(seg_model, classifier, device, long_t, trans_t, 
                   long_shape, trans_shape, resize_target, cls_threshold=0.5,
                   use_tta=False):
    """
    Run inference on a single sample.
    
    Args:
        seg_model: Frozen segmentation model
        classifier: Post-segmentation classifier
        device: torch device
        long_t: [1, H, W] longitudinal image tensor
        trans_t: [1, H, W] transverse image tensor
        long_shape: Original (H, W) of longitudinal image
        trans_shape: Original (H, W) of transverse image
        resize_target: Size to resize to for model input
        cls_threshold: Classification threshold
        use_tta: Whether to use test-time augmentation
    
    Returns:
        pred_long: [H, W] uint8 segmentation mask
        pred_trans: [H, W] uint8 segmentation mask
        cls_pred: int (0 or 1)
        cls_prob: float probability
    """
    # Add batch dimension: [1, 1, H, W]
    xL = long_t.unsqueeze(0).to(device)
    xT = trans_t.unsqueeze(0).to(device)
    
    # Resize to target size
    xL_r = F.interpolate(xL, (resize_target, resize_target), mode="bilinear", align_corners=False)
    xT_r = F.interpolate(xT, (resize_target, resize_target), mode="bilinear", align_corners=False)
    
    # Get segmentation
    segL_logits, segT_logits, _ = seg_model(xL_r, xT_r)
    
    # Get classification
    if use_tta:
        cls_probs = []
        
        # Original
        cls_out = classifier(xL_r, xT_r, segL_logits, segT_logits)
        cls_probs.append(torch.sigmoid(cls_out).cpu().item())
        
        # Horizontal flip
        xL_flip = torch.flip(xL_r, dims=[-1])
        xT_flip = torch.flip(xT_r, dims=[-1])
        segL_flip, segT_flip, _ = seg_model(xL_flip, xT_flip)
        cls_out = classifier(xL_flip, xT_flip, segL_flip, segT_flip)
        cls_probs.append(torch.sigmoid(cls_out).cpu().item())
        
        # Vertical flip
        xL_flip = torch.flip(xL_r, dims=[-2])
        xT_flip = torch.flip(xT_r, dims=[-2])
        segL_flip, segT_flip, _ = seg_model(xL_flip, xT_flip)
        cls_out = classifier(xL_flip, xT_flip, segL_flip, segT_flip)
        cls_probs.append(torch.sigmoid(cls_out).cpu().item())
        
        cls_prob = np.mean(cls_probs)
    else:
        cls_out = classifier(xL_r, xT_r, segL_logits, segT_logits)
        cls_prob = torch.sigmoid(cls_out).cpu().item()
    
    cls_pred = 1 if cls_prob >= cls_threshold else 0
    
    # Upsample segmentation to original sizes
    segL_up = F.interpolate(segL_logits, long_shape, mode="bilinear", align_corners=False)
    segT_up = F.interpolate(segT_logits, trans_shape, mode="bilinear", align_corners=False)
    
    pred_long = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_trans = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    return pred_long, pred_trans, cls_pred, cls_prob


def predict_and_save(seg_model, classifier, device, file_path, long_t, trans_t,
                     long_shape, trans_shape, resize_target, out_dir, 
                     cls_threshold=0.5, use_tta=False):
    """
    Run inference and save results.
    
    Args:
        seg_model: Frozen segmentation model
        classifier: Post-segmentation classifier
        device: torch device
        file_path: Path to input .h5 file
        long_t, trans_t: Image tensors
        long_shape, trans_shape: Original image shapes
        resize_target: Model input size
        out_dir: Output directory
        cls_threshold: Classification threshold
        use_tta: Whether to use TTA
    
    Returns:
        out_path: Path to saved prediction file
        cls_pred: Classification prediction
        cls_prob: Classification probability
    """
    os.makedirs(out_dir, exist_ok=True)
    
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    out_path = os.path.join(out_dir, f"{name_no_ext}_pred.h5")
    
    # Run inference
    pred_long, pred_trans, cls_pred, cls_prob = predict_single(
        seg_model, classifier, device,
        long_t, trans_t, long_shape, trans_shape,
        resize_target, cls_threshold, use_tta
    )
    
    # Save to h5 with same key names as label files
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("long_mask", data=pred_long, compression="gzip")
        hf.create_dataset("trans_mask", data=pred_trans, compression="gzip")
        hf.create_dataset("cls", data=np.array([cls_pred], dtype=np.uint8))
        # Also save probability for potential threshold tuning
        hf.create_dataset("cls_prob", data=np.array([cls_prob], dtype=np.float32))
    
    return out_path, cls_pred, cls_prob


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage inference: Segmentation + Post-Seg Classifier"
    )
    
    # Data paths
    parser.add_argument("--val-dir", type=str, required=True,
                        help="Path to validation folder containing 'images' subfolder with .h5 files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: val-dir/preds)")
    
    # Segmentation model
    parser.add_argument("--seg_checkpoint", type=str, required=True,
                        help="Path to segmentation model checkpoint")
    parser.add_argument("--model", type=str, default="UNetV2",
                        choices=["Echocare", "UNet", "UNetV2"],
                        help="Segmentation model architecture")
    parser.add_argument("--encoder-pth", type=str, default="./pretrain/echocare_encoder.pth",
                        help="Pretrained encoder for Echocare model")
    
    # Classification model
    parser.add_argument("--cls_checkpoint", type=str, required=True,
                        help="Path to post-segmentation classifier checkpoint")
    parser.add_argument("--classifier", type=str, default="full",
                        choices=["full", "seg_only", "seg_img", "light"],
                        help="Classifier variant")
    parser.add_argument("--cls_threshold", type=float, default=None,
                        help="Classification threshold (default: use optimal from checkpoint)")
    
    # Model settings
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize-target", type=int, default=256,
                        help="Input size for model")
    
    # Inference settings
    parser.add_argument("--use_tta", action="store_true",
                        help="Use test-time augmentation")
    parser.add_argument("--gpu", type=str, default="0")
    
    args = parser.parse_args()
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    images_dir = os.path.join(args.val_dir, "images")
    if not os.path.exists(images_dir):
        # Try val-dir directly
        if os.path.exists(args.val_dir) and len(glob.glob(os.path.join(args.val_dir, "*.h5"))) > 0:
            images_dir = args.val_dir
        else:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if args.output_dir is None:
        out_dir = os.path.join(args.val_dir, "preds")
    else:
        out_dir = args.output_dir
    
    print(f"Input directory: {images_dir}")
    print(f"Output directory: {out_dir}")
    
    # Load dataset
    ds = ValH5Dataset(images_dir)
    print(f"Found {len(ds)} files")
    
    # Load models
    print("\n" + "=" * 60)
    seg_model = load_seg_model(args, device)
    print()
    classifier = load_cls_model(args, device)
    print("=" * 60 + "\n")
    
    # Set threshold
    if args.cls_threshold is None:
        args.cls_threshold = getattr(args, 'cls_threshold', 0.5)
    print(f"Classification threshold: {args.cls_threshold:.2f}")
    print(f"TTA enabled: {args.use_tta}")
    print()
    
    # Run inference
    print("Running inference...")
    results = []
    
    for idx in range(len(ds)):
        p, long_t, trans_t, long_shape, trans_shape = ds[idx]
        
        out_path, cls_pred, cls_prob = predict_and_save(
            seg_model, classifier, device,
            p, long_t, trans_t, long_shape, trans_shape,
            args.resize_target, out_dir,
            args.cls_threshold, args.use_tta
        )
        
        results.append({
            "file": os.path.basename(p),
            "cls_pred": cls_pred,
            "cls_prob": cls_prob,
        })
        
        print(f"[{idx+1:3d}/{len(ds)}] {os.path.basename(p)} -> cls={cls_pred} (prob={cls_prob:.3f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    cls_preds = [r["cls_pred"] for r in results]
    cls_probs = [r["cls_prob"] for r in results]
    
    print(f"Total samples: {len(results)}")
    print(f"Predicted vulnerable (cls=1): {sum(cls_preds)} ({100*sum(cls_preds)/len(cls_preds):.1f}%)")
    print(f"Predicted stable (cls=0): {len(cls_preds) - sum(cls_preds)} ({100*(len(cls_preds)-sum(cls_preds))/len(cls_preds):.1f}%)")
    print(f"Mean probability: {np.mean(cls_probs):.3f}")
    print(f"Probability range: [{np.min(cls_probs):.3f}, {np.max(cls_probs):.3f}]")
    print(f"\nPredictions saved to: {out_dir}")
    
    # Save summary CSV
    summary_path = os.path.join(out_dir, "predictions_summary.csv")
    with open(summary_path, "w") as f:
        f.write("filename,cls_pred,cls_prob\n")
        for r in results:
            f.write(f"{r['file']},{r['cls_pred']},{r['cls_prob']:.6f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
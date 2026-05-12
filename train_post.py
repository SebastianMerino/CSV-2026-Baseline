#!/usr/bin/env python3
"""
train_post.py

Two-stage training:
  Stage 1: Load pre-trained segmentation model (frozen)
  Stage 2: Train PostSegClassifier using segmentation outputs

Usage:
    python train_post.py \
        --seg_checkpoint ./checkpoints/best_seg.pth \
        --model UNetV2 \
        --classifier full \
        --train_epochs 100 \
        --gpu 0
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

# Local imports
from dataset.csv import CSVSemiDataset
from model.Echocare import Echocare_UniMatch
from model.unet    import UNetTwoView
from model.unet_v2 import UNetTwoViewV2   # ← new
from model.post_seg_classifier import get_post_classifier


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ensure_cls_shape(y_cls):
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


def unpack_cls(cls_out):
    if isinstance(cls_out, (tuple, list)):
        return cls_out[0]
    return cls_out


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_seg_model(args):
    """Create segmentation model architecture."""
    if args.model == "Echocare":
        model = Echocare_UniMatch(
            img_size=(args.resize_target, args.resize_target),
            in_channels=1,
            seg_out_channels=args.seg_num_classes,
            cls_out_channels=args.cls_num_classes,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            encoder_ckpt=args.echo_care_ckpt,
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


def load_seg_model(args, device, logger):
    """Load and freeze the segmentation model."""
    model = get_seg_model(args)
    
    # Load checkpoint
    if not os.path.exists(args.seg_checkpoint):
        raise FileNotFoundError(f"Segmentation checkpoint not found: {args.seg_checkpoint}")
    
    ckpt = torch.load(args.seg_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded segmentation model from {args.seg_checkpoint}")
    
    if "seg_score" in ckpt:
        logger.info(f"  -> Seg score: {ckpt['seg_score']:.4f}")
    if "cls_score" in ckpt:
        logger.info(f"  -> Cls score: {ckpt['cls_score']:.4f}")
    if "epoch" in ckpt:
        logger.info(f"  -> Trained for {ckpt['epoch']} epochs")
    
    model = model.to(device)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    logger.info(f"Segmentation model frozen ({count_params(model):,} params)")
    
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation for Classification
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationAugment:
    """Simple augmentations for classification training."""
    
    def __init__(self, p_flip=0.5, p_noise=0.3, noise_std=0.05):
        self.p_flip = p_flip
        self.p_noise = p_noise
        self.noise_std = noise_std
    
    def __call__(self, img_long, img_trans):
        # Random horizontal flip (both views together)
        if torch.rand(1).item() < self.p_flip:
            img_long = torch.flip(img_long, dims=[-1])
            img_trans = torch.flip(img_trans, dims=[-1])
        
        # Random noise
        if torch.rand(1).item() < self.p_noise:
            noise_L = torch.randn_like(img_long) * self.noise_std
            noise_T = torch.randn_like(img_trans) * self.noise_std
            img_long = img_long + noise_L
            img_trans = img_trans + noise_T
        
        return img_long, img_trans


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for handling hard examples."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(args, seg_model, classifier, optimizer, train_loader, 
                    device, epoch, logger, augment=None):
    """Train classifier for one epoch."""
    seg_model.eval()
    classifier.train()
    
    if args.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    loss_meter = AverageMeter()
    num_iters = len(train_loader)
    
    for i, (x_long, x_trans, m_long, m_trans, y_cls) in enumerate(train_loader):
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).float().to(device)
        
        # Optional augmentation
        if augment is not None and args.use_augment:
            x_long, x_trans = augment(x_long, x_trans)
        
        # Resize for segmentation model
        x_long_r = F.interpolate(x_long, (args.resize_target, args.resize_target),
                                  mode='bilinear', align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),
                                   mode='bilinear', align_corners=False)
        
        # Get segmentation from frozen model
        with torch.no_grad():
            seg_L, seg_T, _ = seg_model(x_long_r, x_trans_r)
        
        # Forward through classifier
        cls_out = classifier(x_long_r, x_trans_r, seg_L, seg_T)
        
        loss = criterion(cls_out, y_cls)
        
        # Optional: Label smoothing
        if args.label_smoothing > 0:
            smooth_target = y_cls * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            loss = criterion(cls_out, smooth_target)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
        
        optimizer.step()
        
        loss_meter.update(loss.item())
        
        if i % max(1, num_iters // 5) == 0:
            logger.info(
                f"  Iter {i:4d}/{num_iters} | Loss: {loss_meter.avg:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    
    return {"loss": loss_meter.avg}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(args, seg_model, classifier, valid_loader, device, logger, 
             return_details=False):
    """Validate classifier."""
    seg_model.eval()
    classifier.eval()
    
    all_probs = []
    all_preds = []
    all_gts = []
    
    for x_long, x_trans, m_long, m_trans, y_cls in valid_loader:
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        
        # Resize
        x_long_r = F.interpolate(x_long, (args.resize_target, args.resize_target),
                                  mode='bilinear', align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),
                                   mode='bilinear', align_corners=False)
        
        # Segmentation
        seg_L, seg_T, _ = seg_model(x_long_r, x_trans_r)
        
        # Classification
        cls_out = classifier(x_long_r, x_trans_r, seg_L, seg_T)
        
        prob = torch.sigmoid(cls_out).cpu().item()
        pred = 1 if prob >= 0.5 else 0
        gt = int(y_cls.item())
        
        all_probs.append(prob)
        all_preds.append(pred)
        all_gts.append(gt)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    # Metrics at threshold 0.5
    f1 = f1_score(all_gts, all_preds, zero_division=0)
    cm = confusion_matrix(all_gts, all_preds)
    
    # AUC
    try:
        auc = roc_auc_score(all_gts, all_probs)
    except:
        auc = 0.0
    
    # Find optimal threshold
    best_f1, best_thresh = f1, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (all_probs >= t).astype(int)
        f1_t = f1_score(all_gts, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t
    
    best_cm = confusion_matrix(all_gts, (all_probs >= best_thresh).astype(int))
    
    logger.info(f"[Val] F1@0.5: {f1:.4f} | Best F1: {best_f1:.4f} @ thresh={best_thresh:.2f} | AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix @0.5:\n{cm}")
    logger.info(f"Confusion Matrix @{best_thresh:.2f}:\n{best_cm}")
    
    result = {
        "f1": f1,
        "f1_best": best_f1,
        "best_thresh": best_thresh,
        "auc": auc,
        "cm": cm,
        "cm_best": best_cm,
    }
    
    if return_details:
        result["probs"] = all_probs
        result["preds"] = all_preds
        result["gts"] = all_gts
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_with_tta(args, seg_model, classifier, valid_loader, device, logger):
    """Validate with test-time augmentation."""
    seg_model.eval()
    classifier.eval()
    
    all_probs = []
    all_gts = []
    
    for x_long, x_trans, m_long, m_trans, y_cls in valid_loader:
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        
        x_long_r = F.interpolate(x_long, (args.resize_target, args.resize_target),
                                  mode='bilinear', align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),
                                   mode='bilinear', align_corners=False)
        
        probs = []
        
        # Original
        seg_L, seg_T, _ = seg_model(x_long_r, x_trans_r)
        cls_out = classifier(x_long_r, x_trans_r, seg_L, seg_T)
        probs.append(torch.sigmoid(cls_out).cpu().item())
        
        # Horizontal flip
        x_L_flip = torch.flip(x_long_r, dims=[-1])
        x_T_flip = torch.flip(x_trans_r, dims=[-1])
        seg_L, seg_T, _ = seg_model(x_L_flip, x_T_flip)
        cls_out = classifier(x_L_flip, x_T_flip, seg_L, seg_T)
        probs.append(torch.sigmoid(cls_out).cpu().item())
        
        # Vertical flip
        x_L_flip = torch.flip(x_long_r, dims=[-2])
        x_T_flip = torch.flip(x_trans_r, dims=[-2])
        seg_L, seg_T, _ = seg_model(x_L_flip, x_T_flip)
        cls_out = classifier(x_L_flip, x_T_flip, seg_L, seg_T)
        probs.append(torch.sigmoid(cls_out).cpu().item())
        
        # Average
        avg_prob = np.mean(probs)
        all_probs.append(avg_prob)
        all_gts.append(int(y_cls.item()))
    
    all_probs = np.array(all_probs)
    all_gts = np.array(all_gts)
    
    # Find best threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (all_probs >= t).astype(int)
        f1_t = f1_score(all_gts, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t
    
    preds_05 = (all_probs >= 0.5).astype(int)
    f1_05 = f1_score(all_gts, preds_05, zero_division=0)
    
    logger.info(f"[Val+TTA] F1@0.5: {f1_05:.4f} | Best F1: {best_f1:.4f} @ thresh={best_thresh:.2f}")
    
    return {
        "f1": f1_05,
        "f1_best": best_f1,
        "best_thresh": best_thresh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def build_logger(save_path):
    logger = logging.getLogger("TrainPost")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    os.makedirs(save_path, exist_ok=True)

    fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Post-Segmentation Classifier Training")
    
    # Data
    parser.add_argument("--train-labeled-json", type=str, default="./data/train_labeled.json")
    parser.add_argument("--valid-labeled-json", type=str, default="./data/valid.json")
    
    # Segmentation model
    parser.add_argument("--seg_checkpoint", type=str, required=True,
                        help="Path to trained segmentation model checkpoint")
    parser.add_argument("--model", type=str, default="UNetV2", 
                        choices=["Echocare", "UNet", "UNetV2"],
                        help="Segmentation model architecture")
    parser.add_argument("--echo_care_ckpt", type=str, default="./pretrain/echocare_encoder.pth")
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize_target", type=int, default=256)
    
    # Post classifier
    parser.add_argument("--classifier", type=str, default="full",
                        choices=["full", "seg_only", "seg_img", "light"],
                        help="Classifier variant")
    parser.add_argument("--dropout", type=float, default=0.4)
    
    # Training
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "onecycle", "none"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Loss
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # Augmentation
    parser.add_argument("--use_augment", action="store_true", default=True)
    
    # Other
    parser.add_argument("--save_path", type=str, default="./checkpoints_post")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_tta", action="store_true", 
                        help="Use TTA during final evaluation")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save path with timestamp
    args.save_path = os.path.join(args.save_path, f"{args.classifier}")
    
    logger = build_logger(args.save_path)
    logger.info("=" * 70)
    logger.info("Post-Segmentation Classifier Training")
    logger.info("=" * 70)
    logger.info(str(args))
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, "tensorboard"))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load frozen segmentation model
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Loading segmentation model...")
    seg_model = load_seg_model(args, device, logger)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Create classifier
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info(f"Creating classifier: {args.classifier}")
    
    classifier = get_post_classifier(
        variant=args.classifier,
        seg_classes=args.seg_num_classes,
        cls_classes=args.cls_num_classes,
        dropout=args.dropout,
    ).to(device)
    
    logger.info(f"Classifier params: {count_params(classifier):,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Optimizer and scheduler
    # ─────────────────────────────────────────────────────────────────────────
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-6)
    elif args.scheduler == "onecycle":
        # Will be created after dataloader
        scheduler = None
    else:
        scheduler = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Loading data...")
    
    db_train = CSVSemiDataset(args.train_labeled_json, "train_l", size=args.resize_target)
    db_valid = CSVSemiDataset(args.valid_labeled_json, "valid")
    
    train_loader = DataLoader(
        db_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    valid_loader = DataLoader(
        db_valid, batch_size=1, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )
    
    logger.info(f"Train samples: {len(db_train)}")
    logger.info(f"Valid samples: {len(db_valid)}")
    
    # OneCycleLR needs steps_per_epoch
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.train_epochs,
            pct_start=0.1,
        )
    
    # Augmentation
    augment = ClassificationAugment() if args.use_augment else None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initial validation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Initial validation (random classifier):")
    validate(args, seg_model, classifier, valid_loader, device, logger)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    best_f1 = 0.0
    best_f1_opt = 0.0  # Best F1 with optimal threshold
    
    for epoch in range(args.train_epochs):
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch}/{args.train_epochs}")
        
        # Train
        train_stats = train_one_epoch(
            args, seg_model, classifier, optimizer, train_loader,
            device, epoch, logger, augment
        )
        
        # Update scheduler
        if scheduler is not None and args.scheduler == "cosine":
            scheduler.step()
        
        # Validate
        logger.info("-" * 40)
        val_stats = validate(args, seg_model, classifier, valid_loader, device, logger)
        
        # Logging
        writer.add_scalar("Train/Loss", train_stats["loss"], epoch)
        writer.add_scalar("Val/F1", val_stats["f1"], epoch)
        writer.add_scalar("Val/F1_Best", val_stats["f1_best"], epoch)
        writer.add_scalar("Val/AUC", val_stats["auc"], epoch)
        writer.add_scalar("Val/BestThresh", val_stats["best_thresh"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        # Save checkpoints
        is_best = val_stats["f1"] > best_f1
        is_best_opt = val_stats["f1_best"] > best_f1_opt
        
        best_f1 = max(best_f1, val_stats["f1"])
        best_f1_opt = max(best_f1_opt, val_stats["f1_best"])
        
        ckpt = {
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "f1": val_stats["f1"],
            "f1_best": val_stats["f1_best"],
            "best_thresh": val_stats["best_thresh"],
            "auc": val_stats["auc"],
            "args": vars(args),
        }
        
        torch.save(ckpt, os.path.join(args.save_path, "latest.pth"))
        
        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, "best.pth"))
            logger.info(f"★ New best F1@0.5: {val_stats['f1']:.4f}")
        
        if is_best_opt:
            torch.save(ckpt, os.path.join(args.save_path, "best_opt.pth"))
            logger.info(f"★ New best F1@opt: {val_stats['f1_best']:.4f} @ {val_stats['best_thresh']:.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Final evaluation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Final Evaluation")
    logger.info("=" * 70)
    
    # Load best checkpoint
    best_ckpt = torch.load(os.path.join(args.save_path, "best_opt.pth"), map_location="cpu")
    classifier.load_state_dict(best_ckpt["classifier"])
    logger.info(f"Loaded best_opt.pth (epoch {best_ckpt['epoch']})")
    
    # Standard validation
    logger.info("-" * 50)
    logger.info("Standard validation:")
    final_stats = validate(args, seg_model, classifier, valid_loader, device, logger)
    
    # TTA validation
    if args.eval_tta:
        logger.info("-" * 50)
        logger.info("TTA validation:")
        tta_stats = validate_with_tta(args, seg_model, classifier, valid_loader, device, logger)
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Best F1@0.5: {best_f1:.4f}")
    logger.info(f"Best F1@opt: {best_f1_opt:.4f}")
    logger.info(f"Final AUC: {final_stats['auc']:.4f}")
    logger.info(f"Optimal threshold: {final_stats['best_thresh']:.2f}")
    
    writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
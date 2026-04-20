#!/usr/bin/env python3
"""
train_freeze.py

Training script with:
  - Phase 1: Full training (seg + cls) until freeze_epoch
  - Phase 2: Load best_cls.pth, freeze encoder + seg, train only cls head
  - Always tracks and saves best classification model
"""

import os
import sys
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Local imports
# ─────────────────────────────────────────────────────────────────────────────
from dataset.csv import CSVSemiDataset
from util.utils import AverageMeter, count_params, DiceLoss, compute_nsd

from model.Echocare import Echocare_UniMatch
from model.unet    import UNetTwoView
from model.unet_v2 import UNetTwoViewV2   # ← new
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_v2(model):
    m = model.module if isinstance(model, nn.DataParallel) else model
    return isinstance(m, UNetTwoViewV2)


def unpack_cls(cls_out):
    if isinstance(cls_out, (tuple, list)):
        return cls_out[0]
    return cls_out


def ensure_cls_shape(y_cls):
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


def dice_score(pred, gt, smooth=1e-5):
    intersection = (pred & gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(args):
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


# ─────────────────────────────────────────────────────────────────────────────
# Freeze Utilities
# ─────────────────────────────────────────────────────────────────────────────

def freeze_encoder_and_seg(model, logger):
    """Freeze encoder and segmentation decoder parameters."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    frozen_count = 0
    
    for name, param in m.named_parameters():
        should_freeze = False
        
        if isinstance(m, (UNetTwoView, UNetTwoViewV2)):
            if name.startswith("encoder") or name.startswith("seg_decoder"):
                should_freeze = True
        elif isinstance(m, Echocare_UniMatch):
            if (name.startswith("swinViT") or 
                name.startswith("encoder") or 
                name.startswith("decoder_long") or 
                name.startswith("decoder_trans")):
                should_freeze = True
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Frozen {frozen_count} parameter groups (encoder + seg decoders)")
    logger.info(f"Trainable params after freeze: {count_trainable_params(model):,}")


def get_cls_parameters(model):
    """Get only classification head parameters."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    cls_params = []
    
    if isinstance(m, UNetTwoView):
        cls_params.extend(m.cls_fuse.parameters())
    elif isinstance(m, UNetTwoViewV2):
        cls_params.extend(m.ms_embed.parameters())
        cls_params.extend(m.cls_head.parameters())
    elif isinstance(m, Echocare_UniMatch):
        if hasattr(m, "cls_head"):
            cls_params.extend(m.cls_head.parameters())
        if hasattr(m, "cls_fuse"):
            cls_params.extend(m.cls_fuse.parameters())
    
    return cls_params


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-label / CutMix Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def pseudo_from_logits(seg_logits):
    prob = torch.softmax(seg_logits, dim=1)
    conf, mask = prob.max(dim=1)
    return conf, mask


def cutmix_apply_image(img_s, img_mix, box):
    box_ = box.unsqueeze(1).expand_as(img_s)
    out = img_s.clone()
    out[box_ == 1] = img_mix[box_ == 1]
    return out


def cutmix_apply_pseudo(mask, conf, mask_mix, conf_mix, box):
    mask_cm = mask.clone()
    conf_cm = conf.clone()
    mask_cm[box == 1] = mask_mix[box == 1]
    conf_cm[box == 1] = conf_mix[box == 1]
    return mask_cm, conf_cm


# ─────────────────────────────────────────────────────────────────────────────
# Classification Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_cls_loss(criterion, cls_out, y_cls, args):
    if isinstance(cls_out, (tuple, list)):
        logit_fuse, logit_long, logit_trans = cls_out
        loss = (
            criterion(logit_fuse, y_cls) * args.cls_weight_fuse +
            criterion(logit_long, y_cls) * args.cls_weight_long +
            criterion(logit_trans, y_cls) * args.cls_weight_trans
        )
    else:
        loss = criterion(cls_out, y_cls)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training: Full (Seg + Cls)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_full(args, model, optimizer, loader_l, loader_u, loader_u_mix,
                         device, total_iters, epoch, logger, use_amp, amp_dtype, scaler):
    model.train()

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_cls_mse = nn.MSELoss()
    criterion_seg_ce = nn.CrossEntropyLoss()
    criterion_seg_dice = DiceLoss(n_classes=args.seg_num_classes)

    loss_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    mask_ratio_meter = AverageMeter()

    loader = zip(loader_l, loader_u, loader_u_mix)
    num_iters = len(loader_u)

    for i, (
        (x_long, x_trans, m_long, m_trans, y_cls),
        (uL_w, uL_s1, uL_s2, boxL1, boxL2, uT_w, uT_s1, uT_s2, boxT1, boxT2),
        (uL_wm, uL_s1m, uL_s2m, _, _, uT_wm, uT_s1m, uT_s2m, _, _),
    ) in enumerate(loader):

        # To device
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long = m_long.to(device)
        m_trans = m_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        uL_w, uL_s1, uL_s2 = uL_w.to(device), uL_s1.to(device), uL_s2.to(device)
        uT_w, uT_s1, uT_s2 = uT_w.to(device), uT_s1.to(device), uT_s2.to(device)
        boxL1, boxL2 = boxL1.to(device), boxL2.to(device)
        boxT1, boxT2 = boxT1.to(device), boxT2.to(device)
        uL_wm, uL_s1m, uL_s2m = uL_wm.to(device), uL_s1m.to(device), uL_s2m.to(device)
        uT_wm, uT_s1m, uT_s2m = uT_wm.to(device), uT_s1m.to(device), uT_s2m.to(device)

        # Pseudo-labels from weak-mix
        with torch.no_grad():
            model.eval()
            segL_wm, segT_wm, cls_wm_out = model(uL_wm, uT_wm)
            cls_wm = unpack_cls(cls_wm_out).detach()
            confL_wm, maskL_wm = pseudo_from_logits(segL_wm.detach())
            confT_wm, maskT_wm = pseudo_from_logits(segT_wm.detach())

        # CutMix
        uL_s1 = cutmix_apply_image(uL_s1, uL_s1m, boxL1)
        uL_s2 = cutmix_apply_image(uL_s2, uL_s2m, boxL2)
        uT_s1 = cutmix_apply_image(uT_s1, uT_s1m, boxT1)
        uT_s2 = cutmix_apply_image(uT_s2, uT_s2m, boxT2)

        model.train()
        num_l, num_u = x_long.size(0), uL_w.size(0)

        x_long_all = torch.cat([x_long, uL_w], dim=0)
        x_trans_all = torch.cat([x_trans, uT_w], dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            fp_out = model(x_long_all, x_trans_all, need_fp=True)

            if is_v2(model):
                (segL_all, segL_fp_all), (segT_all, segT_fp_all), \
                (cls_f_all, _), (cls_l_all, _), (cls_t_all, _) = fp_out
                segL_x, segL_u_w = segL_all.split([num_l, num_u], dim=0)
                segT_x, segT_u_w = segT_all.split([num_l, num_u], dim=0)
                cls_x_out = (cls_f_all[:num_l], cls_l_all[:num_l], cls_t_all[:num_l])
                cls_u_w_fuse = cls_f_all[num_l:]
                segL_u_w_fp, segT_u_w_fp = segL_fp_all[num_l:], segT_fp_all[num_l:]
            else:
                (segL_all, segL_fp_all), (segT_all, segT_fp_all), (cls_all, _) = fp_out
                segL_x, segL_u_w = segL_all.split([num_l, num_u], dim=0)
                segT_x, segT_u_w = segT_all.split([num_l, num_u], dim=0)
                cls_x_out = cls_all[:num_l]
                cls_u_w_fuse = cls_all[num_l:]
                segL_u_w_fp, segT_u_w_fp = segL_fp_all[num_l:], segT_fp_all[num_l:]

            with torch.no_grad():
                confL_w, maskL_w = pseudo_from_logits(segL_u_w.detach())
                confT_w, maskT_w = pseudo_from_logits(segT_u_w.detach())

            segL_s1, segT_s1, _ = model(uL_s1, uT_s1)
            segL_s2, segT_s2, _ = model(uL_s2, uT_s2)

            maskL_cm1, confL_cm1 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL1)
            maskL_cm2, confL_cm2 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL2)
            maskT_cm1, confT_cm1 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT1)
            maskT_cm2, confT_cm2 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT2)

            # Supervised seg loss
            loss_x_long = (criterion_seg_ce(segL_x, m_long) +
                          criterion_seg_dice(segL_x, m_long, softmax=True,
                                            ignore=torch.zeros_like(m_long))) / 2.0
            loss_x_trans = (criterion_seg_ce(segT_x, m_trans) +
                           criterion_seg_dice(segT_x, m_trans, softmax=True,
                                             ignore=torch.zeros_like(m_trans))) / 2.0
            loss_x_seg = (loss_x_long + loss_x_trans) / 2.0

            # Supervised cls loss
            loss_x_cls = compute_cls_loss(criterion_cls, cls_x_out, y_cls.float(), args)

            # Unsupervised seg loss
            ignL1 = (confL_cm1 < args.conf_thresh).float()
            ignL2 = (confL_cm2 < args.conf_thresh).float()
            ignT1 = (confT_cm1 < args.conf_thresh).float()
            ignT2 = (confT_cm2 < args.conf_thresh).float()

            loss_uL_s = (criterion_seg_dice(segL_s1, maskL_cm1, softmax=True, ignore=ignL1) +
                        criterion_seg_dice(segL_s2, maskL_cm2, softmax=True, ignore=ignL2)) / 2.0
            loss_uT_s = (criterion_seg_dice(segT_s1, maskT_cm1, softmax=True, ignore=ignT1) +
                        criterion_seg_dice(segT_s2, maskT_cm2, softmax=True, ignore=ignT2)) / 2.0
            loss_u_s_seg = (loss_uL_s + loss_uT_s) / 2.0

            # FP loss
            ignLw = (confL_w < args.conf_thresh).float()
            ignTw = (confT_w < args.conf_thresh).float()
            loss_u_w_fp_seg = (
                criterion_seg_dice(segL_u_w_fp, maskL_w, softmax=True, ignore=ignLw) +
                criterion_seg_dice(segT_u_w_fp, maskT_w, softmax=True, ignore=ignTw)
            ) / 2.0

            # Unsupervised cls consistency
            _, _, cls_s1m_out = model(uL_s1m, uT_s1m)
            _, _, cls_s2m_out = model(uL_s2m, uT_s2m)
            cls_s1m = unpack_cls(cls_s1m_out)
            cls_s2m = unpack_cls(cls_s2m_out)
            loss_u_s_cls = (
                criterion_cls_mse(torch.sigmoid(cls_s1m), torch.sigmoid(cls_wm)) +
                criterion_cls_mse(torch.sigmoid(cls_s2m), torch.sigmoid(cls_wm))
            ) / 2.0

            # Total loss
            loss = (
                loss_x_seg * args.seg_loss_weight +
                loss_x_cls * args.cls_loss_weight +
                loss_u_s_seg * args.unsup_seg_weight +
                loss_u_w_fp_seg * args.unsup_seg_weight +
                loss_u_s_cls * args.unsup_cls_weight
            )

        optimizer.zero_grad(set_to_none=True)
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Poly LR
        current_iter = epoch * num_iters + i
        lr = args.base_lr * (1 - current_iter / total_iters) ** 0.9
        optimizer.param_groups[0]["lr"] = lr

        loss_meter.update(loss.item())
        loss_cls_meter.update(loss_x_cls.item())
        mask_ratio = (confL_w >= args.conf_thresh).float().mean()
        mask_ratio_meter.update(mask_ratio.item())

        if i % max(1, num_iters // 8) == 0:
            logger.info(
                f"[FULL] Iter {i:4d}/{num_iters} | Loss: {loss_meter.avg:.3f} | "
                f"Cls: {loss_cls_meter.avg:.3f} | MaskRatio: {mask_ratio_meter.avg:.2f}"
            )

    return {"loss": loss_meter.avg, "loss_cls": loss_cls_meter.avg}


# ─────────────────────────────────────────────────────────────────────────────
# Training: Classification Only (after freeze)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_cls_only(args, model, optimizer, loader_l, device, epoch,
                              logger, use_amp, amp_dtype, scaler):
    model.train()

    criterion_cls = nn.BCEWithLogitsLoss()
    loss_meter = AverageMeter()
    num_iters = len(loader_l)

    for i, (x_long, x_trans, m_long, m_trans, y_cls) in enumerate(loader_l):
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _, _, cls_out = model(x_long, x_trans)
            loss = compute_cls_loss(criterion_cls, cls_out, y_cls.float(), args)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item())

        if i % max(1, num_iters // 8) == 0:
            logger.info(
                f"[CLS-ONLY] Iter {i:4d}/{num_iters} | Loss: {loss_meter.avg:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return {"loss": loss_meter.avg, "loss_cls": loss_meter.avg}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(args, model, valid_loader, device, logger):
    model.eval()

    dice_long = {1: 0.0, 2: 0.0}
    dice_trans = {1: 0.0, 2: 0.0}
    nsd_long = {1: 0.0, 2: 0.0}
    nsd_trans = {1: 0.0, 2: 0.0}
    cls_preds, cls_gts, cls_probs_list = [], [], []

    for x_long, x_trans, m_long, m_trans, y_cls in valid_loader:
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long = m_long.to(device)
        m_trans = m_trans.to(device)

        hL, wL = x_long.shape[-2:]
        hT, wT = x_trans.shape[-2:]

        x_long_r = F.interpolate(x_long, (args.resize_target, args.resize_target),
                                  mode="bilinear", align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),
                                   mode="bilinear", align_corners=False)

        segL, segT, cls_out = model(x_long_r, x_trans_r)
        cls_logit = unpack_cls(cls_out)

        prob = torch.sigmoid(cls_logit).cpu().item()
        pred = 1 if prob >= 0.5 else 0
        gt = int(y_cls.item())

        cls_probs_list.append(prob)
        cls_preds.append(pred)
        cls_gts.append(gt)

        segL = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
        segT = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)
        predL = torch.argmax(segL, dim=1)
        predT = torch.argmax(segT, dim=1)

        for cls in [1, 2]:
            dice_long[cls] += dice_score((predL[0] == cls).cpu().numpy(),
                                          (m_long[0] == cls).cpu().numpy())
            dice_trans[cls] += dice_score((predT[0] == cls).cpu().numpy(),
                                           (m_trans[0] == cls).cpu().numpy())
            nsd_long[cls] += compute_nsd((predL[0] == cls).cpu().numpy(),
                                          (m_long[0] == cls).cpu().numpy(), tolerance=3.0)
            nsd_trans[cls] += compute_nsd((predT[0] == cls).cpu().numpy(),
                                           (m_trans[0] == cls).cpu().numpy(), tolerance=3.0)

    n = len(valid_loader)
    for cls in [1, 2]:
        dice_long[cls] /= n
        dice_trans[cls] /= n
        nsd_long[cls] /= n
        nsd_trans[cls] /= n

    cls_gts = np.array(cls_gts)
    cls_preds = np.array(cls_preds)
    cls_probs_arr = np.array(cls_probs_list)
    
    f1 = f1_score(cls_gts, cls_preds, zero_division=0)
    cm = confusion_matrix(cls_gts, cls_preds)

    # Find optimal threshold
    best_f1, best_thresh = f1, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (cls_probs_arr >= t).astype(int)
        f1_t = f1_score(cls_gts, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    logger.info(f"[Cls] F1@0.5: {f1:.4f} | Best F1: {best_f1:.4f} @ thresh={best_thresh:.2f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    idx_to_name = {1: "Plaque", 2: "Vessel"}
    for cls in [1, 2]:
        logger.info(f"[Dice] {idx_to_name[cls]} | Long: {dice_long[cls]:.3f} | Trans: {dice_trans[cls]:.3f}")
        logger.info(f"[NSD]  {idx_to_name[cls]} | Long: {nsd_long[cls]:.3f} | Trans: {nsd_trans[cls]:.3f}")

    seg_score = (
        (dice_long[2] + nsd_long[2]) / 2 * 0.4 + (dice_long[1] + nsd_long[1]) / 2 * 0.6 +
        (dice_trans[2] + nsd_trans[2]) / 2 * 0.4 + (dice_trans[1] + nsd_trans[1]) / 2 * 0.6
    ) / 2

    total_score = (
        f1 * 0.4 +
        (dice_long[2] + nsd_long[2]) / 2 * 0.4 * 0.2 + (dice_long[1] + nsd_long[1]) / 2 * 0.6 * 0.2 +
        (dice_trans[2] + nsd_trans[2]) / 2 * 0.4 * 0.2 + (dice_trans[1] + nsd_trans[1]) / 2 * 0.6 * 0.2
    )

    return {
        "cls_score": f1,
        "cls_score_best": best_f1,
        "cls_best_thresh": best_thresh,
        "seg_score": seg_score,
        "total_score": total_score,
        "dice_long": dice_long,
        "dice_trans": dice_trans,
        "nsd_long": nsd_long,
        "nsd_trans": nsd_trans,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def build_logger(save_path):
    logger = logging.getLogger("TrainFreeze")
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
    parser = argparse.ArgumentParser("Train with Freeze and Reload Best Cls")
    
    # Data
    parser.add_argument("--train-labeled-json", type=str, default="./data/train_labeled.json")
    parser.add_argument("--train-unlabeled-json", type=str, default="./data/train_unlabeled.json")
    parser.add_argument("--valid-labeled-json", type=str, default="./data/valid.json")

    # Training
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--conf_thresh", type=float, default=0.9)
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize_target", type=int, default=256)

    # Model
    parser.add_argument("--model", type=str, default="UNetV2", choices=["Echocare", "UNet", "UNetV2"])
    parser.add_argument("--echo_care_ckpt", type=str, default="./pretrain/echocare_encoder.pth")
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    # Loss weights
    parser.add_argument("--seg_loss_weight", type=float, default=1.0)
    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--unsup_seg_weight", type=float, default=0.5)
    parser.add_argument("--unsup_cls_weight", type=float, default=0.1)
    parser.add_argument("--cls_weight_fuse", type=float, default=0.5)
    parser.add_argument("--cls_weight_long", type=float, default=0.25)
    parser.add_argument("--cls_weight_trans", type=float, default=0.25)

    # Freeze settings
    parser.add_argument("--freeze_epoch", type=int, default=50,
                        help="Epoch to freeze and reload best_cls. Set -1 to disable.")
    parser.add_argument("--cls_lr", type=float, default=0.0001,
                        help="Learning rate for cls-only phase after freeze.")

    # Paths
    parser.add_argument("--save_path", type=str, default="./checkpoints_freeze")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = build_logger(args.save_path)
    logger.info(str(args))

    cudnn.enabled = True
    cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(args.save_path, "tensorboard"))

    # ─────────────────────────────────────────────────────────────────────────
    # Build model
    # ─────────────────────────────────────────────────────────────────────────
    model = get_model(args).to(device)
    logger.info(f"Model: {args.model} | Params: {count_params(model):,}")

    optimizer = Adam(model.parameters(), lr=args.base_lr)

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # ─────────────────────────────────────────────────────────────────────────
    # Datasets
    # ─────────────────────────────────────────────────────────────────────────
    db_train_u = CSVSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = CSVSemiDataset(args.train_labeled_json, "train_l", size=args.resize_target,
                                 n_sample=len(db_train_u.case_list))
    db_valid_l = CSVSemiDataset(args.valid_labeled_json, "valid")

    train_loader_l = DataLoader(db_train_l, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True)
    train_loader_u = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True)
    train_loader_u_mix = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(db_valid_l, batch_size=1, shuffle=False,
                               num_workers=args.num_workers, drop_last=False, pin_memory=True)

    total_iters = len(train_loader_u) * args.train_epochs

    # ─────────────────────────────────────────────────────────────────────────
    # State tracking
    # ─────────────────────────────────────────────────────────────────────────
    start_epoch = 0
    previous_best = 0.0
    previous_best_seg = 0.0
    previous_best_cls = 0.0
    is_frozen = False

    best_cls_path = os.path.join(args.save_path, "best_cls.pth")
    latest_path = os.path.join(args.save_path, "latest.pth")

    # Resume if exists
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        previous_best = ckpt.get("previous_best", 0.0)
        previous_best_seg = ckpt.get("previous_best_seg", 0.0)
        previous_best_cls = ckpt.get("previous_best_cls", 0.0)
        is_frozen = ckpt.get("is_frozen", False)
        logger.info(f"Resumed from epoch {start_epoch}, best_cls={previous_best_cls:.4f}, frozen={is_frozen}")

        if is_frozen:
            freeze_encoder_and_seg(model, logger)
            optimizer = Adam(get_cls_parameters(model), lr=args.cls_lr)

    # Initial validation
    logger.info("=" * 60)
    logger.info("Initial validation:")
    validate(args, model, valid_loader, device, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.train_epochs):

        # =====================================================================
        # CHECK: Time to freeze and reload best_cls?
        # =====================================================================
        if args.freeze_epoch >= 0 and epoch == args.freeze_epoch and not is_frozen:
            logger.info("=" * 60)
            logger.info(f"FREEZE EPOCH {epoch}: Loading best_cls.pth and freezing encoder + seg")
            logger.info("=" * 60)

            if os.path.exists(best_cls_path):
                ckpt = torch.load(best_cls_path, map_location="cpu")
                model.load_state_dict(ckpt["model"])
                logger.info(f"Loaded best_cls.pth (F1={previous_best_cls:.4f})")
            else:
                logger.warning("best_cls.pth not found! Continuing with current model.")

            # Freeze encoder + seg decoders
            freeze_encoder_and_seg(model, logger)

            # Rebuild optimizer with only cls parameters
            cls_params = get_cls_parameters(model)
            optimizer = Adam(cls_params, lr=args.cls_lr)
            scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
            
            is_frozen = True
            logger.info(f"Optimizer rebuilt with {len(cls_params)} cls parameter groups, LR={args.cls_lr}")

        # =====================================================================
        # TRAINING
        # =====================================================================
        logger.info(f"{'=' * 60}")
        logger.info(f"Epoch {epoch}/{args.train_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Frozen: {is_frozen}")

        if is_frozen:
            stats = train_one_epoch_cls_only(
                args, model, optimizer, train_loader_l, device, epoch,
                logger, use_amp, amp_dtype, scaler
            )
        else:
            stats = train_one_epoch_full(
                args, model, optimizer, train_loader_l, train_loader_u, train_loader_u_mix,
                device, total_iters, epoch, logger, use_amp, amp_dtype, scaler
            )

        # =====================================================================
        # VALIDATION
        # =====================================================================
        logger.info("-" * 40)
        output = validate(args, model, valid_loader, device, logger)

        total_score = output["total_score"]
        seg_score = output["seg_score"]
        cls_score = output["cls_score"]

        writer.add_scalar("Train/Loss", stats["loss"], epoch)
        writer.add_scalar("Train/Loss_Cls", stats["loss_cls"], epoch)
        writer.add_scalar("Val/Total_Score", total_score, epoch)
        writer.add_scalar("Val/Seg_Score", seg_score, epoch)
        writer.add_scalar("Val/Cls_F1", cls_score, epoch)
        writer.add_scalar("Val/Cls_F1_Best", output["cls_score_best"], epoch)
        writer.add_scalar("Val/Cls_Best_Thresh", output["cls_best_thresh"], epoch)
        writer.add_scalar("Val/Is_Frozen", float(is_frozen), epoch)

        # =====================================================================
        # SAVE CHECKPOINTS
        # =====================================================================
        is_best = total_score > previous_best
        is_best_seg = seg_score > previous_best_seg
        is_best_cls = cls_score > previous_best_cls

        previous_best = max(previous_best, total_score)
        previous_best_seg = max(previous_best_seg, seg_score)
        previous_best_cls = max(previous_best_cls, cls_score)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
            "previous_best_seg": previous_best_seg,
            "previous_best_cls": previous_best_cls,
            "is_frozen": is_frozen,
            "cls_score": cls_score,
            "seg_score": seg_score,
            "total_score": total_score,
        }

        torch.save(ckpt, latest_path)

        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, "best.pth"))
            logger.info(f"★ New best total! score={total_score:.4f}")

        if is_best_seg:
            torch.save(ckpt, os.path.join(args.save_path, "best_seg.pth"))
            logger.info(f"★ New best seg! score={seg_score:.4f}")

        if is_best_cls:
            torch.save(ckpt, best_cls_path)
            logger.info(f"★ New best cls! F1={cls_score:.4f}")

    writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
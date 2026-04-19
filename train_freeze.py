#!/usr/bin/env python3
"""
train_freeze.py

Training script with optional freezing of encoder and segmentation decoders
after a specified epoch. After freezing, only the classification head is trained.

Key features:
  - --freeze_epoch: epoch after which encoder + seg decoders are frozen (default: 50)
  - --freeze_epoch -1: disables freezing (trains everything for all epochs)
  - After freezing, only classification loss is computed (no seg losses)
  - Optimizer is rebuilt with only cls parameters after freeze
  - Separate learning rate for cls-only phase via --cls_lr
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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Local imports (adjust paths as needed)
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
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
    """Check if model is UNetTwoViewV2."""
    if isinstance(model, nn.DataParallel):
        return isinstance(model.module, UNetTwoViewV2)
    return isinstance(model, UNetTwoViewV2)


def unpack_cls(cls_out):
    """Extract fused classification logit from v1 or v2 output."""
    if isinstance(cls_out, (tuple, list)):
        return cls_out[0]  # fuse logit
    return cls_out


def ensure_cls_shape(y_cls):
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
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
# Freeze utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_encoder_seg_params(model):
    """Return parameter names belonging to encoder and segmentation decoders."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    encoder_seg_names = []
    
    if isinstance(m, (UNetTwoView, UNetTwoViewV2)):
        # UNet variants: encoder + seg_decoder_long + seg_decoder_trans
        for name, _ in m.named_parameters():
            if name.startswith("encoder") or name.startswith("seg_decoder"):
                encoder_seg_names.append(name)
    elif isinstance(m, Echocare_UniMatch):
        # Echocare: swinViT (encoder) + decoder_long + decoder_trans
        for name, _ in m.named_parameters():
            if (name.startswith("swinViT") or 
                name.startswith("encoder") or 
                name.startswith("decoder_long") or 
                name.startswith("decoder_trans")):
                encoder_seg_names.append(name)
    
    return encoder_seg_names


def get_cls_params(model):
    """Return parameters belonging to classification head only."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    cls_params = []
    
    if isinstance(m, UNetTwoView):
        # cls_fuse
        cls_params.extend(m.cls_fuse.parameters())
    elif isinstance(m, UNetTwoViewV2):
        # ms_embed + cls_head
        cls_params.extend(m.ms_embed.parameters())
        cls_params.extend(m.cls_head.parameters())
    elif isinstance(m, Echocare_UniMatch):
        # cls_head (adjust based on actual EchoCare architecture)
        if hasattr(m, "cls_head"):
            cls_params.extend(m.cls_head.parameters())
        if hasattr(m, "cls_fuse"):
            cls_params.extend(m.cls_fuse.parameters())
    
    return cls_params


def freeze_encoder_and_seg(model, logger):
    """Freeze encoder and segmentation decoder parameters."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    frozen_count = 0
    total_count = 0
    
    encoder_seg_names = get_encoder_seg_params(model)
    
    for name, param in m.named_parameters():
        total_count += 1
        if name in encoder_seg_names:
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Frozen {frozen_count}/{total_count} parameter groups (encoder + seg decoders)")
    logger.info(f"Trainable params after freeze: {count_trainable_params(model):,}")


def rebuild_optimizer_for_cls(model, lr, logger):
    """Create new optimizer with only classification parameters."""
    cls_params = get_cls_params(model)
    
    # Filter to only trainable params
    trainable_cls_params = [p for p in cls_params if p.requires_grad]
    
    if len(trainable_cls_params) == 0:
        logger.warning("No trainable classification parameters found!")
        # Fallback: use all trainable params
        trainable_cls_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = Adam(trainable_cls_params, lr=lr)
    logger.info(f"Rebuilt optimizer for cls-only training with LR={lr}")
    logger.info(f"Number of parameter groups: {len(trainable_cls_params)}")
    
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-label / CutMix helpers
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
# Classification loss (handles both v1 and v2 outputs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cls_loss(criterion, cls_out, y_cls, args, model):
    """
    v1 / Echocare : cls_out is a Tensor → single BCE term.
    v2            : cls_out is (fuse, long, trans) → weighted sum of three terms.
    """
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
# Training: Full (seg + cls)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_full(args, model, optimizer, loader_l, loader_u, loader_u_mix,
                         device, total_iters, epoch, logger, use_amp, amp_dtype, scaler):
    """Full training: both segmentation and classification losses."""
    model.train()

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_cls_mse = nn.MSELoss()
    criterion_seg_ce = nn.CrossEntropyLoss()
    criterion_seg_dice = DiceLoss(n_classes=args.seg_num_classes)

    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_loss_fp = AverageMeter()
    total_loss_extra_cls = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(loader_l, loader_u, loader_u_mix)

    for i, (
        (x_long, x_trans, m_long, m_trans, y_cls),
        (uL_w, uL_s1, uL_s2, boxL1, boxL2, uT_w, uT_s1, uT_s2, boxT1, boxT2),
        (uL_wm, uL_s1m, uL_s2m, _, _, uT_wm, uT_s1m, uT_s2m, _, _),
    ) in enumerate(loader):

        # Move to device
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long = m_long.to(device)
        m_trans = m_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        uL_w = uL_w.to(device)
        uL_s1 = uL_s1.to(device)
        uL_s2 = uL_s2.to(device)
        uT_w = uT_w.to(device)
        uT_s1 = uT_s1.to(device)
        uT_s2 = uT_s2.to(device)
        boxL1 = boxL1.to(device)
        boxL2 = boxL2.to(device)
        boxT1 = boxT1.to(device)
        boxT2 = boxT2.to(device)

        uL_wm = uL_wm.to(device)
        uL_s1m = uL_s1m.to(device)
        uL_s2m = uL_s2m.to(device)
        uT_wm = uT_wm.to(device)
        uT_s1m = uT_s1m.to(device)
        uT_s2m = uT_s2m.to(device)

        # Step 1: pseudo-labels from weak-mix (no grad)
        with torch.no_grad():
            model.eval()
            segL_wm, segT_wm, cls_wm_out = model(uL_wm, uT_wm)
            cls_wm = unpack_cls(cls_wm_out).detach()
            confL_wm, maskL_wm = pseudo_from_logits(segL_wm.detach())
            confT_wm, maskT_wm = pseudo_from_logits(segT_wm.detach())

        # Step 2: CutMix strong images
        uL_s1 = cutmix_apply_image(uL_s1, uL_s1m, boxL1)
        uL_s2 = cutmix_apply_image(uL_s2, uL_s2m, boxL2)
        uT_s1 = cutmix_apply_image(uT_s1, uT_s1m, boxT1)
        uT_s2 = cutmix_apply_image(uT_s2, uT_s2m, boxT2)

        model.train()
        num_l_bs = x_long.size(0)
        num_u_bs = uL_w.size(0)

        # Step 3: joint forward — labeled + weak unlabeled (with fp)
        x_long_all = torch.cat([x_long, uL_w], dim=0)
        x_trans_all = torch.cat([x_trans, uT_w], dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            fp_out = model(x_long_all, x_trans_all, need_fp=True)

            if is_v2(model):
                (segL_all, segL_fp_all), (segT_all, segT_fp_all), \
                (cls_f_all, cls_f_fp_all), (cls_l_all, _), (cls_t_all, _) = fp_out

                segL_x, segL_u_w = segL_all.split([num_l_bs, num_u_bs], dim=0)
                segT_x, segT_u_w = segT_all.split([num_l_bs, num_u_bs], dim=0)

                cls_x_out = (
                    cls_f_all.split([num_l_bs, num_u_bs], dim=0)[0],
                    cls_l_all.split([num_l_bs, num_u_bs], dim=0)[0],
                    cls_t_all.split([num_l_bs, num_u_bs], dim=0)[0],
                )
                cls_u_w_fuse = cls_f_all.split([num_l_bs, num_u_bs], dim=0)[1]

                segL_u_w_fp = segL_fp_all[num_l_bs:]
                segT_u_w_fp = segT_fp_all[num_l_bs:]
            else:
                (segL_all, segL_fp_all), (segT_all, segT_fp_all), (cls_all, cls_fp_all) = fp_out

                segL_x, segL_u_w = segL_all.split([num_l_bs, num_u_bs], dim=0)
                segT_x, segT_u_w = segT_all.split([num_l_bs, num_u_bs], dim=0)
                cls_x_out = cls_all.split([num_l_bs, num_u_bs], dim=0)[0]
                cls_u_w_fuse = cls_all.split([num_l_bs, num_u_bs], dim=0)[1]

                segL_u_w_fp = segL_fp_all[num_l_bs:]
                segT_u_w_fp = segT_fp_all[num_l_bs:]

            # Step 4: pseudo-labels from weak unlabeled
            with torch.no_grad():
                confL_w, maskL_w = pseudo_from_logits(segL_u_w.detach())
                confT_w, maskT_w = pseudo_from_logits(segT_u_w.detach())

            # Step 5: strong forward + CutMix pseudo
            segL_s1, segT_s1, _ = model(uL_s1, uT_s1)
            segL_s2, segT_s2, _ = model(uL_s2, uT_s2)

            maskL_cm1, confL_cm1 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL1)
            maskL_cm2, confL_cm2 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL2)
            maskT_cm1, confT_cm1 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT1)
            maskT_cm2, confT_cm2 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT2)

            # Step 6: losses
            # Supervised segmentation
            loss_x_long = (criterion_seg_ce(segL_x, m_long) +
                          criterion_seg_dice(segL_x, m_long, softmax=True,
                                            ignore=torch.zeros_like(m_long))) / 2.0
            loss_x_trans = (criterion_seg_ce(segT_x, m_trans) +
                           criterion_seg_dice(segT_x, m_trans, softmax=True,
                                             ignore=torch.zeros_like(m_trans))) / 2.0
            loss_x_seg = (loss_x_long + loss_x_trans) / 2.0

            # Supervised classification
            loss_x_cls = compute_cls_loss(criterion_cls, cls_x_out, y_cls.float(), args, model)

            # Unsupervised segmentation consistency
            ignL1 = (confL_cm1 < args.conf_thresh).float()
            ignL2 = (confL_cm2 < args.conf_thresh).float()
            ignT1 = (confT_cm1 < args.conf_thresh).float()
            ignT2 = (confT_cm2 < args.conf_thresh).float()

            loss_uL_s = (criterion_seg_dice(segL_s1, maskL_cm1, softmax=True, ignore=ignL1) +
                        criterion_seg_dice(segL_s2, maskL_cm2, softmax=True, ignore=ignL2)) / 2.0
            loss_uT_s = (criterion_seg_dice(segT_s1, maskT_cm1, softmax=True, ignore=ignT1) +
                        criterion_seg_dice(segT_s2, maskT_cm2, softmax=True, ignore=ignT2)) / 2.0
            loss_u_s_seg = (loss_uL_s + loss_uT_s) / 2.0

            # Feature-perturbation consistency
            ignLw = (confL_w < args.conf_thresh).float()
            ignTw = (confT_w < args.conf_thresh).float()
            loss_u_w_fp_seg = (
                criterion_seg_dice(segL_u_w_fp, maskL_w, softmax=True, ignore=ignLw) +
                criterion_seg_dice(segT_u_w_fp, maskT_w, softmax=True, ignore=ignTw)
            ) / 2.0

            # Unsupervised classification consistency
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
                loss_x_seg + loss_x_cls +
                loss_u_s_seg * 0.5 +
                loss_u_w_fp_seg * 0.5 +
                loss_u_s_cls * 0.1
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
        iters = epoch * len(loader_u) + i
        lr = args.base_lr * (1 - iters / total_iters) ** 0.9
        optimizer.param_groups[0]["lr"] = lr

        total_loss.update(loss.item())
        total_loss_x.update(loss_x_seg.item() + loss_x_cls.item())
        total_loss_s.update(loss_u_s_seg.item())
        total_loss_fp.update(loss_u_w_fp_seg.item())
        total_loss_extra_cls.update(loss_u_s_cls.item())
        mask_ratio = (confL_w >= args.conf_thresh).sum() / confL_w.numel()
        total_mask_ratio.update(mask_ratio.item())

        if i % max(1, len(loader_u) // 8) == 0:
            logger.info(
                f"Iters: {i:4d} | Total: {total_loss.avg:.3f} | "
                f"Loss_x: {total_loss_x.avg:.3f} | Loss_s: {total_loss_s.avg:.3f} | "
                f"Loss_fp: {total_loss_fp.avg:.3f} | "
                f"Loss_extra_cls: {total_loss_extra_cls.avg:.3f} | "
                f"MaskRatio: {total_mask_ratio.avg:.3f}"
            )

    return {
        "loss": total_loss.avg,
        "loss_x": total_loss_x.avg,
        "loss_s": total_loss_s.avg,
        "loss_fp": total_loss_fp.avg,
        "loss_extra_cls": total_loss_extra_cls.avg,
        "mask_ratio": total_mask_ratio.avg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training: Classification only (after freeze)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_cls_only(args, model, optimizer, loader_l, device, epoch,
                             logger, use_amp, amp_dtype, scaler, total_iters_cls):
    """
    Classification-only training phase.
    Only uses labeled data and classification loss.
    Encoder and segmentation decoders are frozen.
    """
    model.train()

    criterion_cls = nn.BCEWithLogitsLoss()
    total_loss = AverageMeter()

    for i, (x_long, x_trans, m_long, m_trans, y_cls) in enumerate(loader_l):
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # Forward pass (encoder is frozen, so no grads flow through it)
            _, _, cls_out = model(x_long, x_trans)

            # Classification loss only
            loss = compute_cls_loss(criterion_cls, cls_out, y_cls.float(), args, model)

            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # Constant LR or cosine annealing for cls-only phase
        # Using constant LR here; can be changed to cosine
        iters = (epoch - args.freeze_epoch) * len(loader_l) + i
        # Optional: cosine annealing
        # lr = args.cls_lr * 0.5 * (1 + np.cos(np.pi * iters / total_iters_cls))
        # optimizer.param_groups[0]["lr"] = lr

        total_loss.update(loss.item())

        if i % max(1, len(loader_l) // 8) == 0:
            logger.info(
                f"[CLS-ONLY] Iters: {i:4d} | Loss: {total_loss.avg:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return {
        "loss": total_loss.avg,
        "loss_x": total_loss.avg,
        "loss_s": 0.0,
        "loss_fp": 0.0,
        "loss_extra_cls": 0.0,
        "mask_ratio": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(args, model, valid_loader, device, logger, writer=None, epoch=None):
    model.eval()

    dice_long = {1: 0.0, 2: 0.0}
    dice_trans = {1: 0.0, 2: 0.0}
    nsd_long = {1: 0.0, 2: 0.0}
    nsd_trans = {1: 0.0, 2: 0.0}
    cls_pred_list = []
    cls_gt_list = []

    num_batches = len(valid_loader)

    for (x_long, x_trans, m_long, m_trans, y_cls) in valid_loader:
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long = m_long.to(device)
        m_trans = m_trans.to(device)
        y_cls = y_cls.to(device)

        hL, wL = x_long.shape[-2:]
        hT, wT = x_trans.shape[-2:]

        x_long_r = F.interpolate(x_long, (args.resize_target, args.resize_target),
                                  mode="bilinear", align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),
                                   mode="bilinear", align_corners=False)

        segL, segT, cls_out = model(x_long_r, x_trans_r)
        cls_logit = unpack_cls(cls_out)

        cls_prob = torch.sigmoid(cls_logit)
        cls_pred = (cls_prob >= 0.5).long().view(-1)
        cls_pred_list.extend(cls_pred.cpu().numpy().tolist())
        cls_gt_list.extend(y_cls.view(-1).cpu().numpy().tolist())

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

    idx_to_name = {1: "Plaque", 2: "Vessel"}
    for cls in [1, 2]:
        dice_long[cls] /= max(1, num_batches)
        dice_trans[cls] /= max(1, num_batches)
        nsd_long[cls] /= max(1, num_batches)
        nsd_trans[cls] /= max(1, num_batches)
        logger.info(f"[Dice] {idx_to_name[cls]} | Long: {dice_long[cls]:.2f} | Trans: {dice_trans[cls]:.2f}")
        logger.info(f"[NSD]  {idx_to_name[cls]} | Long: {nsd_long[cls]:.2f} | Trans: {nsd_trans[cls]:.2f}")

    mean_dice = (dice_long[1] + dice_long[2] + dice_trans[1] + dice_trans[2]) / 4.0
    mean_NSD = (nsd_long[1] + nsd_long[2] + nsd_trans[1] + nsd_trans[2]) / 4.0
    logger.info(f"[Dice] Mean Foreground: {mean_dice:.3f}")
    logger.info(f"[NSD]  Mean Foreground: {mean_NSD:.3f}")

    cls_gt = np.array(cls_gt_list)
    cls_pred = np.array(cls_pred_list)
    f1 = f1_score(cls_gt, cls_pred, zero_division=0)
    logger.info(f"[Cls] F1 Score: {f1:.4f}")

    cm = confusion_matrix(cls_gt, cls_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

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
        "dice_long_vessel": dice_long[2],
        "dice_long_plaque": dice_long[1],
        "dice_trans_vessel": dice_trans[2],
        "dice_trans_plaque": dice_trans[1],
        "nsd_long_vessel": nsd_long[2],
        "nsd_long_plaque": nsd_long[1],
        "nsd_trans_vessel": nsd_trans[2],
        "nsd_trans_plaque": nsd_trans[1],
        "cls_score": f1,
        "seg_score": seg_score,
        "total_score": total_score,
    }


def dice_score(pred, gt, smooth=1e-5):
    """Compute Dice score between two binary masks."""
    intersection = (pred & gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def build_logger(save_path: str):
    logger = logging.getLogger("UniMatch TwoView Training (Freeze)")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    os.makedirs(save_path, exist_ok=True)

    fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
    fh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("UniMatch Two-View Training with Freeze")
    parser.add_argument("--train-labeled-json", type=str, default="./data/train_labeled.json")
    parser.add_argument("--train-unlabeled-json", type=str, default="./data/train_unlabeled.json")
    parser.add_argument("--valid-labeled-json", type=str, default="./data/valid.json")

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--conf_thresh", type=float, default=0.9)
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize_target", type=int, default=256)

    parser.add_argument("--echo_care_ckpt", type=str, default="./pretrain/echocare_encoder.pth")
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    # Model choice
    parser.add_argument(
        "--model", type=str, default="UNetV2",
        choices=["Echocare", "UNet", "UNetV2"],
        help="Echocare: SwinUNETR-based | UNet: baseline | UNetV2: improved cls head"
    )

    # Classification loss weights (for UNetV2)
    parser.add_argument("--cls_weight_fuse", type=float, default=0.5)
    parser.add_argument("--cls_weight_long", type=float, default=0.25)
    parser.add_argument("--cls_weight_trans", type=float, default=0.25)

    # Freeze settings
    parser.add_argument(
        "--freeze_epoch", type=int, default=50,
        help="Epoch after which to freeze encoder + seg decoders. Set to -1 to disable."
    )
    parser.add_argument(
        "--cls_lr", type=float, default=0.0001,
        help="Learning rate for cls-only training phase after freeze."
    )

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

    tb_logdir = os.path.join(args.save_path, "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)
    logger.info(f"TensorBoard log dir: {tb_logdir}")

    # Build model
    model = get_model(args)
    logger.info(f"Model: {args.model}")
    logger.info(f"Total params: {count_params(model):,}")
    model = model.to(device)

    # Optimizer (full model initially)
    optimizer = Adam(model.parameters(), lr=args.base_lr)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))

    # Datasets
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

    # Resume
    previous_best = 0.0
    previous_best_seg = 0.0
    previous_best_cls = 0.0
    start_epoch = 0
    is_frozen = False

    latest_ckpt = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        previous_best = ckpt.get("previous_best", 0.0)
        previous_best_seg = ckpt.get("previous_best_seg", 0.0)
        previous_best_cls = ckpt.get("previous_best_cls", 0.0)
        is_frozen = ckpt.get("is_frozen", False)
        logger.info(f"Resumed from {latest_ckpt}, epoch={start_epoch}, best={previous_best:.2f}, frozen={is_frozen}")

        # Re-apply freeze if we're past freeze_epoch
        if is_frozen or (args.freeze_epoch >= 0 and start_epoch > args.freeze_epoch):
            freeze_encoder_and_seg(model, logger)
            optimizer = rebuild_optimizer_for_cls(model, args.cls_lr, logger)
            is_frozen = True

    # Calculate total iters for cls-only phase
    if args.freeze_epoch >= 0:
        total_iters_cls = len(train_loader_l) * (args.train_epochs - args.freeze_epoch)
    else:
        total_iters_cls = 0

    # Initial validation
    validate(args, model, valid_loader, device, logger, writer=writer, epoch=0)

    # Training loop
    for epoch in range(start_epoch, args.train_epochs):
        # Check if we should freeze
        if args.freeze_epoch >= 0 and epoch == args.freeze_epoch and not is_frozen:
            logger.info("=" * 60)
            logger.info(f"FREEZING ENCODER + SEG DECODERS at epoch {epoch}")
            logger.info("=" * 60)
            freeze_encoder_and_seg(model, logger)
            optimizer = rebuild_optimizer_for_cls(model, args.cls_lr, logger)
            scaler = torch.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))
            is_frozen = True

        logger.info(
            f"===========> Epoch: {epoch}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Previous best: {previous_best:.2f}, "
            f"Frozen: {is_frozen}"
        )

        if is_frozen:
            # Classification-only training
            stats = train_one_epoch_cls_only(
                args=args, model=model, optimizer=optimizer,
                loader_l=train_loader_l, device=device, epoch=epoch,
                logger=logger, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
                total_iters_cls=total_iters_cls,
            )
        else:
            # Full training (seg + cls)
            stats = train_one_epoch_full(
                args=args, model=model, optimizer=optimizer,
                loader_l=train_loader_l, loader_u=train_loader_u,
                loader_u_mix=train_loader_u_mix,
                device=device, total_iters=total_iters, epoch=epoch,
                logger=logger, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
            )

        # TensorBoard logging
        writer.add_scalar("Train/Total_Loss", stats["loss"], epoch)
        writer.add_scalar("Train/Loss_x", stats["loss_x"], epoch)
        writer.add_scalar("Train/Loss_s", stats["loss_s"], epoch)
        writer.add_scalar("Train/Loss_fp", stats["loss_fp"], epoch)
        writer.add_scalar("Train/Loss_extra_cls", stats["loss_extra_cls"], epoch)
        writer.add_scalar("Train/Is_Frozen", float(is_frozen), epoch)

        # Validation
        output_dict = validate(args, model, valid_loader, device, logger, writer=writer, epoch=epoch)

        writer.add_scalar("Val/Dice/Long_Vessel", output_dict["dice_long_vessel"], epoch)
        writer.add_scalar("Val/Dice/Long_Plaque", output_dict["dice_long_plaque"], epoch)
        writer.add_scalar("Val/Dice/Trans_Vessel", output_dict["dice_trans_vessel"], epoch)
        writer.add_scalar("Val/Dice/Trans_Plaque", output_dict["dice_trans_plaque"], epoch)
        writer.add_scalar("Val/NSD/Long_Vessel", output_dict["nsd_long_vessel"], epoch)
        writer.add_scalar("Val/NSD/Long_Plaque", output_dict["nsd_long_plaque"], epoch)
        writer.add_scalar("Val/NSD/Trans_Vessel", output_dict["nsd_trans_vessel"], epoch)
        writer.add_scalar("Val/NSD/Trans_Plaque", output_dict["nsd_trans_plaque"], epoch)
        writer.add_scalar("Val/Cls/F1_Mean", output_dict["cls_score"], epoch)
        writer.add_scalar("Val/Total_Score", output_dict["total_score"], epoch)

        total_score = output_dict["total_score"]
        seg_score = output_dict.get("seg_score", 0.0)
        cls_score = output_dict.get("cls_score", 0.0)

        is_best = total_score > previous_best
        is_best_seg = seg_score > previous_best_seg
        is_best_cls = cls_score > previous_best_cls

        previous_best = max(previous_best, total_score)
        previous_best_seg = max(previous_best_seg, seg_score)
        previous_best_cls = max(previous_best_cls, cls_score)

        # Save checkpoints
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
            "previous_best_seg": previous_best_seg,
            "previous_best_cls": previous_best_cls,
            "is_frozen": is_frozen,
        }
        torch.save(ckpt, latest_ckpt)

        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, "best.pth"))
            logger.info(f"New best! total_score={total_score:.4f} saved to best.pth")

        if is_best_seg:
            torch.save(ckpt, os.path.join(args.save_path, "best_seg.pth"))
            logger.info(f"New best segmentation! seg_score={seg_score:.4f}")

        if is_best_cls:
            torch.save(ckpt, os.path.join(args.save_path, "best_cls.pth"))
            logger.info(f"New best classification! cls_score={cls_score:.4f}")

    writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
import argparse
import logging
import os
import sys
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset.csv import CSVSemiDataset
from util.utils import AverageMeter, count_params, DiceLoss, compute_nsd

from model.Echocare import Echocare_UniMatch
from model.unet    import UNetTwoView
from model.unet_v2 import UNetTwoViewV2   # ← new


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared between models
# ─────────────────────────────────────────────────────────────────────────────

def is_v2(model) -> bool:
    """True when the model is UNetTwoViewV2 (returns 3-tuple cls logits)."""
    return isinstance(model, UNetTwoViewV2)


def unpack_cls(cls_out):
    """
    Normalise classifier output so the rest of the code always gets a
    single fused logit tensor, regardless of model version.

      v1 / Echocare : cls_out is Tensor [B, 1]  → returned as-is
      v2            : cls_out is (fuse, long, trans) → return fuse tensor
    """
    if isinstance(cls_out, (tuple, list)):
        return cls_out[0]          # fused logit
    return cls_out


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative grid (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

SEG_PALETTE = {0: None, 1: (1.0, 0.2, 0.2), 2: (0.2, 0.9, 0.2)}
SEG_ALPHA   = 0.45


def _norm_image(arr):
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)


def _overlay_mask(base_grey, mask):
    rgb = np.stack([base_grey, base_grey, base_grey], axis=-1)
    for cls_idx, colour in SEG_PALETTE.items():
        if colour is None:
            continue
        region = mask == cls_idx
        if not region.any():
            continue
        for c in range(3):
            rgb[..., c][region] = rgb[..., c][region] * (1 - SEG_ALPHA) + colour[c] * SEG_ALPHA
    return rgb


def save_qualitative_grid(args, model, valid_loader, device, epoch, save_path):
    model.eval()
    all_batches = list(valid_loader)
    n_samples   = min(4, len(all_batches))
    chosen_idx  = np.random.choice(len(all_batches), size=n_samples, replace=False)

    samples = []
    with torch.no_grad():
        for idx in chosen_idx:
            x_long, x_trans, m_long, m_trans, y_cls = all_batches[idx]
            x_long  = x_long.to(device)
            x_trans = x_trans.to(device)

            hL, wL = x_long.shape[-2:]
            hT, wT = x_trans.shape[-2:]
            xL_r = F.interpolate(x_long,  (args.resize_target, args.resize_target), mode="bilinear", align_corners=False)
            xT_r = F.interpolate(x_trans, (args.resize_target, args.resize_target), mode="bilinear", align_corners=False)

            segL, segT, cls_out = model(xL_r, xT_r)

            segL = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
            segT = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)

            cls_logit = unpack_cls(cls_out)
            cls_prob  = torch.sigmoid(cls_logit).item()
            cls_pred  = int(cls_prob >= 0.5)
            cls_gt    = int(y_cls.view(-1)[0].item())

            samples.append({
                "xL": x_long[0, 0].cpu().numpy(), "xT": x_trans[0, 0].cpu().numpy(),
                "gtL": m_long[0].cpu().numpy(),   "gtT": m_trans[0].cpu().numpy(),
                "predL": torch.argmax(segL, dim=1)[0].cpu().numpy(),
                "predT": torch.argmax(segT, dim=1)[0].cpu().numpy(),
                "cls_pred": cls_pred, "cls_prob": cls_prob, "cls_gt": cls_gt,
            })

    n_rows, fig_w, fig_h = 6, n_samples * 3.0, n_samples * 2.8
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    if n_samples == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    row_titles = ["Long — Image", "Long — GT", "Long — Pred",
                  "Trans — Image", "Trans — GT", "Trans — Pred"]

    for col, s in enumerate(samples):
        xL_n, xT_n = _norm_image(s["xL"]), _norm_image(s["xT"])
        panels = [
            np.stack([xL_n, xL_n, xL_n], axis=-1),
            _overlay_mask(xL_n, s["gtL"]),
            _overlay_mask(xL_n, s["predL"]),
            np.stack([xT_n, xT_n, xT_n], axis=-1),
            _overlay_mask(xT_n, s["gtT"]),
            _overlay_mask(xT_n, s["predT"]),
        ]
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=7, rotation=0, labelpad=72, va="center")

        gt_str   = "Vulnerable" if s["cls_gt"]   == 1 else "Non-vuln"
        pred_str = "Vulnerable" if s["cls_pred"]  == 1 else "Non-vuln"
        match    = "✓" if s["cls_pred"] == s["cls_gt"] else "✗"
        colour   = "#44dd88" if s["cls_pred"] == s["cls_gt"] else "#ff5555"
        axes[n_rows - 1, col].set_xlabel(
            f"GT: {gt_str}\nPred: {pred_str} ({s['cls_prob']:.2f}) {match}",
            fontsize=7, color=colour, labelpad=4)

    fig.legend(handles=[mpatches.Patch(color=SEG_PALETTE[1], label="Plaque"),
                        mpatches.Patch(color=SEG_PALETTE[2], label="Vessel")],
               loc="lower center", ncol=2, fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Validation samples — Epoch {epoch}  (new best)", fontsize=10, y=1.002)
    plt.tight_layout()

    grid_dir = os.path.join(save_path, "qualitative_grids")
    os.makedirs(grid_dir, exist_ok=True)
    out_path = os.path.join(grid_dir, f"epoch_{epoch:04d}_best.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("UniMatch Two-View Training")
    parser.add_argument("--train-labeled-json",   type=str, default="./data/train_labeled.json")
    parser.add_argument("--train-unlabeled-json", type=str, default="./data/train_unlabeled.json")
    parser.add_argument("--valid-labeled-json",   type=str, default="./data/valid.json")

    parser.add_argument("--train_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--base_lr",        type=float, default=0.0001)
    parser.add_argument("--conf_thresh",    type=float, default=0.9)
    parser.add_argument("--seg_num_classes",type=int,   default=3)
    parser.add_argument("--cls_num_classes",type=int,   default=1)
    parser.add_argument("--resize_target",  type=int,   default=256)

    parser.add_argument("--echo_care_ckpt", type=str, default="./pretrain/echocare_encoder.pth")
    parser.add_argument("--amp",      type=bool, default=True)
    parser.add_argument("--amp-dtype",type=str,  default="fp16", choices=["fp16", "bf16"])

    # ── model choice ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="Echocare",
        choices=["Echocare", "UNet", "UNetV2"],
        help=(
            "Echocare : SwinUNETR-based (best overall)\n"
            "UNet     : lightweight baseline\n"
            "UNetV2   : lightweight + multi-scale cls head (improved classification)"
        ),
    )

    # ── UNetV2 classification loss weights ────────────────────────────────────
    parser.add_argument(
        "--cls_weight_fuse",  type=float, default=0.5,
        help="Weight for the fused-view classification loss (UNetV2 only).")
    parser.add_argument(
        "--cls_weight_long",  type=float, default=0.25,
        help="Weight for the longitudinal-view classification loss (UNetV2 only).")
    parser.add_argument(
        "--cls_weight_trans", type=float, default=0.25,
        help="Weight for the transverse-view classification loss (UNetV2 only).")

    parser.add_argument("--save_path",   type=str, default="./checkpoints_new_arq")
    parser.add_argument("--gpu",         type=str, default="3")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = build_logger(args.save_path)
    logger.info(str(args))

    cudnn.enabled   = True
    cudnn.benchmark = True

    tb_logdir = os.path.join(args.save_path, "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)
    logger.info(f"TensorBoard log dir: {tb_logdir}")

    model = get_model(args)
    logger.info("Total params: {:.1f}M".format(count_params(model)))
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.base_lr)

    use_amp   = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler    = torch.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))

    db_train_u = CSVSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = CSVSemiDataset(args.train_labeled_json,   "train_l", size=args.resize_target,
                                n_sample=len(db_train_u.case_list))
    db_valid_l = CSVSemiDataset(args.valid_labeled_json,   "valid")

    train_loader_l     = DataLoader(db_train_l, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True)
    train_loader_u     = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True)
    train_loader_u_mix = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True)
    valid_loader       = DataLoader(db_valid_l, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False, pin_memory=True)

    total_iters = len(train_loader_u) * args.train_epochs

    previous_best     = 0.0
    previous_best_seg = 0.0
    previous_best_cls = 0.0
    start_epoch       = 0

    latest_ckpt = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch       = ckpt["epoch"] + 1
        previous_best     = ckpt.get("previous_best",     0.0)
        previous_best_seg = ckpt.get("previous_best_seg", 0.0)
        previous_best_cls = ckpt.get("previous_best_cls", 0.0)
        logger.info(f"Resumed from {latest_ckpt}, epoch={start_epoch}, best={previous_best:.2f}")

    validate(args, model, valid_loader, device, logger, writer=writer, epoch=0)

    for epoch in range(start_epoch, args.train_epochs):
        logger.info(
            f"===========> Epoch: {epoch}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Previous best: {previous_best:.2f}"
        )

        stats = train_one_epoch(
            args=args, model=model, optimizer=optimizer,
            loader_l=train_loader_l, loader_u=train_loader_u,
            loader_u_mix=train_loader_u_mix,
            device=device, total_iters=total_iters, epoch=epoch,
            logger=logger, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
        )

        writer.add_scalar("Train/Total_Loss",      stats["loss"],           epoch)
        writer.add_scalar("Train/Loss_x",          stats["loss_x"],         epoch)
        writer.add_scalar("Train/Loss_s",          stats["loss_s"],         epoch)
        writer.add_scalar("Train/Loss_fp",         stats["loss_fp"],        epoch)
        writer.add_scalar("Train/Loss_extra_cls",  stats["loss_extra_cls"], epoch)

        output_dict = validate(args, model, valid_loader, device, logger, writer=writer, epoch=epoch)

        writer.add_scalar("Val/Dice/Long_Vessel",  output_dict["dice_long_vessel"],  epoch)
        writer.add_scalar("Val/Dice/Long_Plaque",  output_dict["dice_long_plaque"],  epoch)
        writer.add_scalar("Val/Dice/Trans_Vessel", output_dict["dice_trans_vessel"], epoch)
        writer.add_scalar("Val/Dice/Trans_Plaque", output_dict["dice_trans_plaque"], epoch)
        writer.add_scalar("Val/NSD/Long_Vessel",   output_dict["nsd_long_vessel"],   epoch)
        writer.add_scalar("Val/NSD/Long_Plaque",   output_dict["nsd_long_plaque"],   epoch)
        writer.add_scalar("Val/NSD/Trans_Vessel",  output_dict["nsd_trans_vessel"],  epoch)
        writer.add_scalar("Val/NSD/Trans_Plaque",  output_dict["nsd_trans_plaque"],  epoch)
        writer.add_scalar("Val/Cls/F1_Mean",       output_dict["cls_score"],         epoch)
        writer.add_scalar("Val/Total_Score",       output_dict["total_score"],       epoch)

        total_score = output_dict["total_score"]
        seg_score   = output_dict.get("seg_score", 0.0)
        cls_score   = output_dict.get("cls_score", 0.0)

        is_best     = total_score > previous_best
        is_best_seg = seg_score   > previous_best_seg
        is_best_cls = cls_score   > previous_best_cls

        previous_best     = max(previous_best,     total_score)
        previous_best_seg = max(previous_best_seg, seg_score)
        previous_best_cls = max(previous_best_cls, cls_score)

        ckpt = {
            "model":              model.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "epoch":              epoch,
            "previous_best":      previous_best,
            "previous_best_seg":  previous_best_seg,
            "previous_best_cls":  previous_best_cls,
        }
        torch.save(ckpt, latest_ckpt)

        if epoch%5==1:
            grid_path = save_qualitative_grid(args, model, valid_loader, device, epoch, args.save_path)
            logger.info(f"Qualitative grid saved to {grid_path}")

        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, "best.pth"))
            logger.info(f"New best! total_score={total_score:.2f}")
            grid_path = save_qualitative_grid(args, model, valid_loader, device, epoch, args.save_path)
            logger.info(f"Qualitative grid saved to {grid_path}")

        if is_best_seg:
            torch.save(ckpt, os.path.join(args.save_path, "best_seg.pth"))
            logger.info(f"New best segmentation! seg_score={seg_score:.4f}")

        if is_best_cls:
            torch.save(ckpt, os.path.join(args.save_path, "best_cls.pth"))
            logger.info(f"New best classification! cls_score={cls_score:.4f}")

    writer.close()
    logger.info("Training finished.")


# ─────────────────────────────────────────────────────────────────────────────
# Classification loss  (handles both v1 and v2 outputs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cls_loss(criterion, cls_out, y_cls, args, model):
    """
    v1 / Echocare : cls_out is a Tensor → single BCE term.
    v2            : cls_out is (fuse, long, trans) → weighted sum of three terms.
                    Weights are controlled by --cls_weight_fuse/long/trans.
    """
    if isinstance(cls_out, (tuple, list)):
        # UNetV2 path
        logit_fuse, logit_long, logit_trans = cls_out
        loss = (
            criterion(logit_fuse,  y_cls) * args.cls_weight_fuse  +
            criterion(logit_long,  y_cls) * args.cls_weight_long  +
            criterion(logit_trans, y_cls) * args.cls_weight_trans
        )
    else:
        loss = criterion(cls_out, y_cls)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-label / CutMix helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def pseudo_from_logits(seg_logits):
    prob = torch.softmax(seg_logits, dim=1)
    conf, mask = prob.max(dim=1)
    return conf, mask


def cutmix_apply_image(img_s, img_mix, box):
    box_ = box.unsqueeze(1).expand_as(img_s)
    out  = img_s.clone()
    out[box_ == 1] = img_mix[box_ == 1]
    return out


def cutmix_apply_pseudo(mask, conf, mask_mix, conf_mix, box):
    mask_cm = mask.clone(); conf_cm = conf.clone()
    mask_cm[box == 1] = mask_mix[box == 1]
    conf_cm[box == 1] = conf_mix[box == 1]
    return mask_cm, conf_cm


def ensure_cls_shape(y_cls):
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


# ─────────────────────────────────────────────────────────────────────────────
# train_one_epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(args, model, optimizer, loader_l, loader_u, loader_u_mix,
                    device, total_iters, epoch, logger, use_amp, amp_dtype, scaler):
    model.train()

    criterion_cls     = nn.BCEWithLogitsLoss()
    criterion_cls_mse = nn.MSELoss()
    criterion_seg_ce  = nn.CrossEntropyLoss()
    criterion_seg_dice= DiceLoss(n_classes=args.seg_num_classes)

    total_loss          = AverageMeter()
    total_loss_x        = AverageMeter()
    total_loss_s        = AverageMeter()
    total_loss_fp       = AverageMeter()
    total_loss_extra_cls= AverageMeter()
    total_mask_ratio    = AverageMeter()

    loader = zip(loader_l, loader_u, loader_u_mix)

    for i, (
        (x_long, x_trans, m_long, m_trans, y_cls),
        (uL_w, uL_s1, uL_s2, boxL1, boxL2, uT_w, uT_s1, uT_s2, boxT1, boxT2),
        (uL_wm, uL_s1m, uL_s2m, _, _, uT_wm, uT_s1m, uT_s2m, _, _),
    ) in enumerate(loader):

        x_long  = x_long.to(device);  x_trans = x_trans.to(device)
        m_long  = m_long.to(device);  m_trans = m_trans.to(device)
        y_cls   = ensure_cls_shape(y_cls).to(device)

        uL_w  = uL_w.to(device);  uL_s1 = uL_s1.to(device); uL_s2 = uL_s2.to(device)
        uT_w  = uT_w.to(device);  uT_s1 = uT_s1.to(device); uT_s2 = uT_s2.to(device)
        boxL1 = boxL1.to(device); boxL2 = boxL2.to(device)
        boxT1 = boxT1.to(device); boxT2 = boxT2.to(device)

        uL_wm  = uL_wm.to(device);  uL_s1m = uL_s1m.to(device); uL_s2m = uL_s2m.to(device)
        uT_wm  = uT_wm.to(device);  uT_s1m = uT_s1m.to(device); uT_s2m = uT_s2m.to(device)

        # ── Step 1: pseudo-labels from weak-mix (no grad) ─────────────────────
        with torch.no_grad():
            model.eval()
            segL_wm, segT_wm, cls_wm_out = model(uL_wm, uT_wm)
            cls_wm = unpack_cls(cls_wm_out).detach()
            confL_wm, maskL_wm = pseudo_from_logits(segL_wm.detach())
            confT_wm, maskT_wm = pseudo_from_logits(segT_wm.detach())

        # ── Step 2: CutMix strong images ──────────────────────────────────────
        uL_s1 = cutmix_apply_image(uL_s1, uL_s1m, boxL1)
        uL_s2 = cutmix_apply_image(uL_s2, uL_s2m, boxL2)
        uT_s1 = cutmix_apply_image(uT_s1, uT_s1m, boxT1)
        uT_s2 = cutmix_apply_image(uT_s2, uT_s2m, boxT2)

        model.train()
        num_l_bs = x_long.size(0)
        num_u_bs = uL_w.size(0)

        # ── Step 3: joint forward — labeled + weak unlabeled (with fp) ────────
        x_long_all  = torch.cat([x_long, uL_w], dim=0)
        x_trans_all = torch.cat([x_trans, uT_w], dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):

            fp_out = model(x_long_all, x_trans_all, need_fp=True)

            if is_v2(model):
                # UNetV2 returns 5-tuple:
                # (seg_l, seg_l_fp), (seg_t, seg_t_fp),
                # (cls_f, cls_f_fp), (cls_l, cls_l_fp), (cls_t, cls_t_fp)
                (segL_all, segL_fp_all), (segT_all, segT_fp_all), \
                (cls_f_all, cls_f_fp_all), \
                (cls_l_all, _), (cls_t_all, _) = fp_out

                # split labeled / unlabeled
                segL_x,  segL_u_w  = segL_all.split([num_l_bs, num_u_bs], dim=0)
                segT_x,  segT_u_w  = segT_all.split([num_l_bs, num_u_bs], dim=0)

                # cls output for labeled split — keep as tuple so compute_cls_loss can unpack
                cls_x_out = (
                    cls_f_all.split([num_l_bs, num_u_bs], dim=0)[0],
                    cls_l_all.split([num_l_bs, num_u_bs], dim=0)[0],
                    cls_t_all.split([num_l_bs, num_u_bs], dim=0)[0],
                )
                cls_u_w_fuse = cls_f_all.split([num_l_bs, num_u_bs], dim=0)[1]  # for pseudo later

                segL_u_w_fp = segL_fp_all[num_l_bs:]
                segT_u_w_fp = segT_fp_all[num_l_bs:]

            else:
                # Echocare / UNetV1: 3-tuple of pairs
                (segL_all, segL_fp_all), (segT_all, segT_fp_all), (cls_all, cls_fp_all) = fp_out

                segL_x,  segL_u_w  = segL_all.split([num_l_bs, num_u_bs], dim=0)
                segT_x,  segT_u_w  = segT_all.split([num_l_bs, num_u_bs], dim=0)
                cls_x_out  = cls_all.split([num_l_bs, num_u_bs], dim=0)[0]
                cls_u_w_fuse = cls_all.split([num_l_bs, num_u_bs], dim=0)[1]

                segL_u_w_fp = segL_fp_all[num_l_bs:]
                segT_u_w_fp = segT_fp_all[num_l_bs:]

            # ── Step 4: strong forward (s1 + s2 batched) ──────────────────────
            uL_s = torch.cat([uL_s1, uL_s2], dim=0)
            uT_s = torch.cat([uT_s1, uT_s2], dim=0)
            segL_s_out, segT_s_out, _ = model(uL_s, uT_s)
            segL_s1, segL_s2 = segL_s_out.chunk(2, dim=0)
            segT_s1, segT_s2 = segT_s_out.chunk(2, dim=0)

            # ── Step 5: CutMix pseudo-labels ───────────────────────────────────
            confL_w, maskL_w = pseudo_from_logits(segL_u_w.detach())
            confT_w, maskT_w = pseudo_from_logits(segT_u_w.detach())

            maskL_cm1, confL_cm1 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL1)
            maskL_cm2, confL_cm2 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL2)
            maskT_cm1, confT_cm1 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT1)
            maskT_cm2, confT_cm2 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT2)

            # ── Step 6: losses ─────────────────────────────────────────────────

            # — supervised segmentation —
            loss_x_long  = (criterion_seg_ce(segL_x, m_long) +
                            criterion_seg_dice(segL_x, m_long, softmax=True,
                                              ignore=torch.zeros_like(m_long))) / 2.0
            loss_x_trans = (criterion_seg_ce(segT_x, m_trans) +
                            criterion_seg_dice(segT_x, m_trans, softmax=True,
                                              ignore=torch.zeros_like(m_trans))) / 2.0
            loss_x_seg   = (loss_x_long + loss_x_trans) / 2.0

            # — supervised classification —
            # compute_cls_loss handles both (fuse,long,trans) tuple and plain tensor
            loss_x_cls = compute_cls_loss(criterion_cls, cls_x_out, y_cls.float(), args, model)

            # — unsupervised segmentation consistency —
            ignL1 = (confL_cm1 < args.conf_thresh).float()
            ignL2 = (confL_cm2 < args.conf_thresh).float()
            ignT1 = (confT_cm1 < args.conf_thresh).float()
            ignT2 = (confT_cm2 < args.conf_thresh).float()

            loss_uL_s    = (criterion_seg_dice(segL_s1, maskL_cm1, softmax=True, ignore=ignL1) +
                            criterion_seg_dice(segL_s2, maskL_cm2, softmax=True, ignore=ignL2)) / 2.0
            loss_uT_s    = (criterion_seg_dice(segT_s1, maskT_cm1, softmax=True, ignore=ignT1) +
                            criterion_seg_dice(segT_s2, maskT_cm2, softmax=True, ignore=ignT2)) / 2.0
            loss_u_s_seg = (loss_uL_s + loss_uT_s) / 2.0

            # — feature-perturbation consistency —
            ignLw = (confL_w < args.conf_thresh).float()
            ignTw = (confT_w < args.conf_thresh).float()
            loss_u_w_fp_seg = (
                criterion_seg_dice(segL_u_w_fp, maskL_w, softmax=True, ignore=ignLw) +
                criterion_seg_dice(segT_u_w_fp, maskT_w, softmax=True, ignore=ignTw)
            ) / 2.0

            # — unsupervised classification consistency —
            # Compare strong-mix cls outputs to weak-mix cls_wm using the fused
            # logit only (CutMix labels are ambiguous for per-view heads).
            _, _, cls_s1m_out = model(uL_s1m, uT_s1m)
            _, _, cls_s2m_out = model(uL_s2m, uT_s2m)
            cls_s1m = unpack_cls(cls_s1m_out)
            cls_s2m = unpack_cls(cls_s2m_out)
            loss_u_s_cls = (
                criterion_cls_mse(torch.sigmoid(cls_s1m), torch.sigmoid(cls_wm)) +
                criterion_cls_mse(torch.sigmoid(cls_s2m), torch.sigmoid(cls_wm))
            ) / 2.0

            # — total —
            loss = (
                loss_x_seg
                + loss_x_cls
                + loss_u_s_seg   * 0.5
                + loss_u_w_fp_seg* 0.5
                + loss_u_s_cls   * 0.1
            )

            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # poly LR
        iters = epoch * len(loader_u) + i
        lr    = args.base_lr * (1 - iters / total_iters) ** 0.9
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
        "loss": total_loss.avg, "loss_x": total_loss_x.avg,
        "loss_s": total_loss_s.avg, "loss_fp": total_loss_fp.avg,
        "loss_extra_cls": total_loss_extra_cls.avg, "mask_ratio": total_mask_ratio.avg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# validate  (unchanged logic — unpack_cls handles v2 tuple)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(args, model, valid_loader, device, logger, writer=None, epoch=None):
    model.eval()

    dice_long  = {1: 0.0, 2: 0.0}
    dice_trans = {1: 0.0, 2: 0.0}
    nsd_long   = {1: 0.0, 2: 0.0}
    nsd_trans  = {1: 0.0, 2: 0.0}
    cls_pred_list = []
    cls_gt_list   = []

    num_batches = len(valid_loader)
    val_idx     = 0

    for (x_long, x_trans, m_long, m_trans, y_cls) in valid_loader:
        x_long  = x_long.to(device);  x_trans = x_trans.to(device)
        m_long  = m_long.to(device);  m_trans = m_trans.to(device)
        y_cls   = y_cls.to(device)

        hL, wL = x_long.shape[-2:]
        hT, wT = x_trans.shape[-2:]

        xL_r = F.interpolate(x_long,  (args.resize_target, args.resize_target), mode="bilinear", align_corners=False)
        xT_r = F.interpolate(x_trans, (args.resize_target, args.resize_target), mode="bilinear", align_corners=False)

        segL, segT, cls_out = model(xL_r, xT_r)

        # ── classification ─────────────────────────────────────────────────────
        cls_logit = unpack_cls(cls_out)           # always a plain Tensor
        cls_prob  = torch.sigmoid(cls_logit)
        cls_pred  = (cls_prob >= 0.5).long().view(-1)
        cls_pred_list.extend(cls_pred.cpu().numpy().tolist())
        cls_gt_list.extend(y_cls.view(-1).cpu().numpy().tolist())

        # ── segmentation ───────────────────────────────────────────────────────
        segL = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
        segT = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)
        predL = torch.argmax(segL, dim=1)
        predT = torch.argmax(segT, dim=1)

        # ── TensorBoard visualisation ──────────────────────────────────────────
        if writer is not None:
            try:
                xL_n = (lambda im: (im - im.min()) / max(im.max() - im.min(), 1e-8))(x_long[0,0].cpu().numpy())
                xT_n = (lambda im: (im - im.min()) / max(im.max() - im.min(), 1e-8))(x_trans[0,0].cpu().numpy())
                predL_np = predL[0].cpu().numpy(); gtL_np = m_long[0].cpu().numpy()
                predT_np = predT[0].cpu().numpy(); gtT_np = m_trans[0].cpu().numpy()
                baseL = np.stack([xL_n]*3, 0); baseT = np.stack([xT_n]*3, 0)

                def overlay(base, mask, color, alpha=0.5):
                    out = base.copy()
                    for c in range(3):
                        out[c][mask] = out[c][mask]*(1-alpha) + color[c]*alpha
                    return out

                red, green = [1.,0.,0.], [0.,1.,0.]
                step = epoch if (epoch is not None and epoch >= 0) else 0
                writer.add_image(f"Val/vis/long/{val_idx}",
                    torch.from_numpy(np.concatenate([
                        baseL,
                        overlay(overlay(baseL.copy(), predL_np==1, red), predL_np==2, green),
                        overlay(overlay(baseL.copy(), gtL_np==1,   red), gtL_np==2,   green),
                    ], axis=2)).float(), global_step=step)
                writer.add_image(f"Val/vis/trans/{val_idx}",
                    torch.from_numpy(np.concatenate([
                        baseT,
                        overlay(overlay(baseT.copy(), predT_np==1, red), predT_np==2, green),
                        overlay(overlay(baseT.copy(), gtT_np==1,   red), gtT_np==2,   green),
                    ], axis=2)).float(), global_step=step)
            except Exception as e:
                logger.warning(f"Failed to write val visualisation: {e}")

        val_idx += 1

        for cls in [1, 2]:
            interL = ((predL==cls) & (m_long==cls)).sum().item()
            unionL = (predL==cls).sum().item() + (m_long==cls).sum().item()
            dice_long[cls]  += 2.0 * interL / (unionL + 1e-8)

            interT = ((predT==cls) & (m_trans==cls)).sum().item()
            unionT = (predT==cls).sum().item() + (m_trans==cls).sum().item()
            dice_trans[cls] += 2.0 * interT / (unionT + 1e-8)

            nsd_long[cls]  += compute_nsd((predL[0]==cls).cpu().numpy(), (m_long[0]==cls).cpu().numpy(),  tolerance=3.0)
            nsd_trans[cls] += compute_nsd((predT[0]==cls).cpu().numpy(), (m_trans[0]==cls).cpu().numpy(), tolerance=3.0)

    idx_to_name = {1: "Plaque", 2: "Vessel"}
    for cls in [1, 2]:
        dice_long[cls]  /= max(1, num_batches)
        dice_trans[cls] /= max(1, num_batches)
        nsd_long[cls]   /= max(1, num_batches)
        nsd_trans[cls]  /= max(1, num_batches)
        logger.info(f"[Dice] {idx_to_name[cls]} | Long: {dice_long[cls]:.2f} | Trans: {dice_trans[cls]:.2f}")
        logger.info(f"[NSD]  {idx_to_name[cls]} | Long: {nsd_long[cls]:.2f} | Trans: {nsd_trans[cls]:.2f}")

    mean_dice = (dice_long[1]+dice_long[2]+dice_trans[1]+dice_trans[2]) / 4.0
    mean_NSD  = (nsd_long[1] +nsd_long[2] +nsd_trans[1]+nsd_trans[2])  / 4.0
    logger.info(f"[Dice] Mean Foreground: {mean_dice:.3f}")
    logger.info(f"[NSD]  Mean Foreground: {mean_NSD:.3f}")

    cls_gt   = np.array(cls_gt_list)
    cls_pred = np.array(cls_pred_list)
    f1       = f1_score(cls_gt, cls_pred, zero_division=0)
    logger.info(f"[Cls] F1 Score: {f1:.4f}")

    cm = confusion_matrix(cls_gt, cls_pred)
    def _fmt_cm(cm_a, lbls):
        lines = ["\t" + "\t".join(f"Pred:{l}" for l in lbls)]
        for i, l in enumerate(lbls):
            rt = cm_a[i].sum()
            pct = "\t".join(f"{cm_a[i,j]/rt*100:.1f}%" for j in range(cm_a.shape[1])) if rt else "n/a"
            lines.append(f"True:{l}\t{chr(9).join(str(int(x)) for x in cm_a[i])}\t| {pct}")
        return "\n".join(lines)
    logger.info("Confusion Matrix:\n" + _fmt_cm(cm, list(range(cm.shape[0]))))

    return {
        "dice_long_vessel":  dice_long[2],  "dice_long_plaque":  dice_long[1],
        "dice_trans_vessel": dice_trans[2], "dice_trans_plaque": dice_trans[1],
        "nsd_long_vessel":   nsd_long[2],   "nsd_long_plaque":   nsd_long[1],
        "nsd_trans_vessel":  nsd_trans[2],  "nsd_trans_plaque":  nsd_trans[1],
        "cls_score": f1,
        "seg_score": (
            (dice_long[2]+nsd_long[2])/2*0.4 + (dice_long[1]+nsd_long[1])/2*0.6 +
            (dice_trans[2]+nsd_trans[2])/2*0.4 + (dice_trans[1]+nsd_trans[1])/2*0.6
        ) / 2,
        "total_score": (
            f1*0.4 +
            (dice_long[2]+nsd_long[2])/2*0.4*0.2 + (dice_long[1]+nsd_long[1])/2*0.6*0.2 +
            (dice_trans[2]+nsd_trans[2])/2*0.4*0.2 + (dice_trans[1]+nsd_trans[1])/2*0.6*0.2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_logger(save_path):
    logger = logging.getLogger("UniMatch TwoView Training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    os.makedirs(save_path, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    fh  = logging.FileHandler(os.path.join(save_path, "log.txt"))
    sh  = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def get_model(args):
    """
    Model factory.

    --model Echocare  →  SwinUNETR-based (best overall performance)
    --model UNet      →  lightweight baseline (original)
    --model UNetV2    →  lightweight + multi-scale cls head (improved classification)
    """
    if args.model == "Echocare":
        return Echocare_UniMatch(
            in_chns=1, seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes, encoder_pth=args.echo_care_ckpt,
        )
    if args.model == "UNet":
        return UNetTwoView(
            in_chns=1, seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    if args.model == "UNetV2":
        return UNetTwoViewV2(
            in_chns=1, seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
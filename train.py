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
matplotlib.use("Agg")  # non-interactive backend — safe for training scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset.csv import CSVSemiDataset
from util.utils import AverageMeter, count_params, DiceLoss, compute_nsd

from model.Echocare import Echocare_UniMatch
from model.unet import UNetTwoView


# ------------------------------------------------------------------
# Colour palette for segmentation overlays
#   class 0 = background  (transparent)
#   class 1 = Plaque      (red)
#   class 2 = Vessel      (green)
# ------------------------------------------------------------------
SEG_PALETTE = {
    0: None,                  # background — no overlay
    1: (1.0, 0.2, 0.2),      # plaque  — red
    2: (0.2, 0.9, 0.2),      # vessel  — green
}
SEG_ALPHA = 0.45              # overlay transparency


def _norm_image(arr: np.ndarray) -> np.ndarray:
    """Normalise a 2-D float array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _overlay_mask(base_grey: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend a segmentation mask onto a greyscale image.

    Parameters
    ----------
    base_grey : H×W float in [0, 1]
    mask      : H×W int with class indices

    Returns
    -------
    H×W×3 float RGB in [0, 1]
    """
    rgb = np.stack([base_grey, base_grey, base_grey], axis=-1)  # H,W,3
    for cls_idx, colour in SEG_PALETTE.items():
        if colour is None:
            continue
        region = mask == cls_idx
        if not region.any():
            continue
        for c in range(3):
            rgb[..., c][region] = (
                rgb[..., c][region] * (1 - SEG_ALPHA) + colour[c] * SEG_ALPHA
            )
    return rgb


def save_qualitative_grid(
    args,
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    epoch: int,
    save_path: str,
) -> None:
    """
    Pick up to 4 random validation samples, run inference, and save a
    PNG grid.  Each column shows one sample; rows are:

        Row 0 — Longitudinal input image
        Row 1 — Long  GT  mask overlay
        Row 2 — Long  Pred mask overlay
        Row 3 — Transverse input image
        Row 4 — Trans GT  mask overlay
        Row 5 — Trans Pred mask overlay

    A text annotation on each column shows the GT class label and the
    predicted class (with confidence).
    """
    model.eval()

    # ----------------------------------------------------------------
    # Collect up to 4 random samples from the validation loader
    # ----------------------------------------------------------------
    all_batches = list(valid_loader)          # list of single-sample batches
    n_total = len(all_batches)
    n_samples = min(4, n_total)
    chosen_idx = np.random.choice(n_total, size=n_samples, replace=False)

    samples = []
    with torch.no_grad():
        for idx in chosen_idx:
            x_long, x_trans, m_long, m_trans, y_cls = all_batches[idx]

            x_long  = x_long.to(device)
            x_trans = x_trans.to(device)

            # resize for model
            hL, wL = x_long.shape[-2:]
            hT, wT = x_trans.shape[-2:]
            xL_r = F.interpolate(x_long,  (args.resize_target, args.resize_target),
                                 mode="bilinear", align_corners=False)
            xT_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),
                                 mode="bilinear", align_corners=False)

            segL, segT, cls_out = model(xL_r, xT_r)

            # resize predictions back to original resolution
            segL = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
            segT = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)

            predL = torch.argmax(segL, dim=1)[0].cpu().numpy()   # H,W
            predT = torch.argmax(segT, dim=1)[0].cpu().numpy()

            cls_prob  = torch.sigmoid(cls_out).item()
            cls_pred  = int(cls_prob >= 0.5)
            cls_gt    = int(y_cls.view(-1)[0].item())

            samples.append({
                "xL":       x_long[0, 0].cpu().numpy(),    # H,W
                "xT":       x_trans[0, 0].cpu().numpy(),
                "gtL":      m_long[0].cpu().numpy(),        # H,W int
                "gtT":      m_trans[0].cpu().numpy(),
                "predL":    predL,
                "predT":    predT,
                "cls_pred": cls_pred,
                "cls_prob": cls_prob,
                "cls_gt":   cls_gt,
            })

    # ----------------------------------------------------------------
    # Build the figure:  6 rows × n_samples columns
    # ----------------------------------------------------------------
    n_rows = 6
    fig_w = n_samples * 3.0
    fig_h = n_rows * 2.8
    fig, axes = plt.subplots(n_rows, n_samples,
                             figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})

    # Ensure axes is always 2-D even for n_samples == 1
    if n_samples == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    row_titles = [
        "Long — Image",
        "Long — GT",
        "Long — Pred",
        "Trans — Image",
        "Trans — GT",
        "Trans — Pred",
    ]

    for col, s in enumerate(samples):
        xL_n  = _norm_image(s["xL"])
        xT_n  = _norm_image(s["xT"])

        panels = [
            np.stack([xL_n, xL_n, xL_n], axis=-1),          # row 0: long image
            _overlay_mask(xL_n, s["gtL"]),                   # row 1: long GT
            _overlay_mask(xL_n, s["predL"]),                 # row 2: long pred
            np.stack([xT_n, xT_n, xT_n], axis=-1),          # row 3: trans image
            _overlay_mask(xT_n, s["gtT"]),                   # row 4: trans GT
            _overlay_mask(xT_n, s["predT"]),                 # row 5: trans pred
        ]

        for row, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label on the leftmost column
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=7,
                              rotation=0, labelpad=72, va="center")

        # ---- per-column classification annotation (bottom of column) ----
        ax_bottom = axes[n_rows - 1, col]
        gt_str   = "Vulnerable" if s["cls_gt"]   == 1 else "Non-vuln"
        pred_str = "Vulnerable" if s["cls_pred"]  == 1 else "Non-vuln"
        match    = "✓" if s["cls_pred"] == s["cls_gt"] else "✗"
        colour   = "#44dd88" if s["cls_pred"] == s["cls_gt"] else "#ff5555"

        ax_bottom.set_xlabel(
            f"GT: {gt_str}\nPred: {pred_str} ({s['cls_prob']:.2f}) {match}",
            fontsize=7,
            color=colour,
            labelpad=4,
        )

    # ---- global legend for segmentation colours ----
    legend_patches = [
        mpatches.Patch(color=SEG_PALETTE[1], label="Plaque"),
        mpatches.Patch(color=SEG_PALETTE[2], label="Vessel"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(f"Validation samples — Epoch {epoch}  (new best)", fontsize=10, y=1.002)
    plt.tight_layout()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    grid_dir = os.path.join(save_path, "qualitative_grids")
    os.makedirs(grid_dir, exist_ok=True)
    out_path = os.path.join(grid_dir, f"epoch_{epoch:04d}_best.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser("UniMatch Two-View Training")
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
    parser.add_argument('--amp', type=bool, default=True, help='enable torch.cuda.amp')
    parser.add_argument('--amp-dtype', type=str, default='fp16', choices=['fp16', 'bf16'])

    # model choice: Echocare (SwinUNETR-based) or UNet
    parser.add_argument("--model", type=str, default="Echocare", choices=["Echocare", "UNet"],
                        help="Model architecture to use: 'Echocare' or 'UNet'")

    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--gpu", type=str, default="3")
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

    model = get_model(args)

    logger.info("Total params: {:.1f}M".format(count_params(model)))
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.base_lr)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))

    db_train_u = CSVSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = CSVSemiDataset(args.train_labeled_json, "train_l", size=args.resize_target, n_sample=len(db_train_u.case_list))
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

    # resume
    previous_best = 0.0
    start_epoch = 0
    previous_best_seg = 0.0
    previous_best_cls = 0.0
    latest_ckpt = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        previous_best = ckpt.get("previous_best", 0.0)
        previous_best_seg = ckpt.get("previous_best_seg", 0.0)
        previous_best_cls = ckpt.get("previous_best_cls", 0.0)
        logger.info(f"************ Resume from {latest_ckpt}, epoch={start_epoch}, best={previous_best:.2f}")

    output_dict = validate(args, model, valid_loader, device, logger, writer=writer, epoch=0)

    for epoch in range(start_epoch, args.train_epochs):
        logger.info(f"===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}, Previous best: {previous_best:.2f}")

        stats = train_one_epoch(
            args=args,
            model=model,
            optimizer=optimizer,
            loader_l=train_loader_l,
            loader_u=train_loader_u,
            loader_u_mix=train_loader_u_mix,
            device=device,
            total_iters=total_iters,
            epoch=epoch,
            logger=logger,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler
        )

        writer.add_scalar("Train/Total_Loss", stats["loss"], epoch)
        writer.add_scalar("Train/Loss_x", stats["loss_x"], epoch)
        writer.add_scalar("Train/Loss_s", stats["loss_s"], epoch)
        writer.add_scalar("Train/Loss_fp", stats["loss_fp"], epoch)
        writer.add_scalar("Train/Loss_extra_cls", stats["loss_extra_cls"], epoch)

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

        is_best = total_score > previous_best
        previous_best = max(previous_best, total_score)
        seg_score = output_dict.get("seg_score", 0.0)
        cls_score = output_dict.get("cls_score", 0.0)
        is_best_seg = seg_score > previous_best_seg
        is_best_cls = cls_score > previous_best_cls
        previous_best_seg = max(previous_best_seg, seg_score)
        previous_best_cls = max(previous_best_cls, cls_score)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
        }
        torch.save(ckpt, latest_ckpt)
        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, "best.pth"))
            logger.info(f"New best! total_score={total_score:.2f} saved to best.pth")

        # ---- save qualitative grid ----
        grid_path = save_qualitative_grid(
            args, model, valid_loader, device, epoch, args.save_path
        )
        logger.info(f"Qualitative grid saved to {grid_path}")

        if is_best_seg:
            torch.save(ckpt, os.path.join(args.save_path, "best_seg.pth"))
            logger.info(f"New best segmentation! seg_score={seg_score:.4f} saved to best_seg.pth")
        if is_best_cls:
            torch.save(ckpt, os.path.join(args.save_path, "best_cls.pth"))
            logger.info(f"New best classification! cls_score={cls_score:.4f} saved to best_cls.pth")
        # always save latest previous_best values into latest_ckpt for resume
        ckpt["previous_best_seg"] = previous_best_seg
        ckpt["previous_best_cls"] = previous_best_cls

    writer.close()
    logger.info("Training finished.")


@torch.no_grad()
def pseudo_from_logits(seg_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seg_logits: [B,K,H,W]
    returns:
      conf: [B,H,W] (max softmax prob)
      mask: [B,H,W] (argmax)
    """
    prob = torch.softmax(seg_logits, dim=1)
    conf, mask = prob.max(dim=1)
    return conf, mask


def cutmix_apply_image(img_s: torch.Tensor, img_mix: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """
    img_s/img_mix: [B,1,H,W], box: [B,H,W] 0/1
    """
    box_ = box.unsqueeze(1).expand_as(img_s)
    out = img_s.clone()
    out[box_ == 1] = img_mix[box_ == 1]
    return out


def cutmix_apply_pseudo(mask: torch.Tensor, conf: torch.Tensor,
                        mask_mix: torch.Tensor, conf_mix: torch.Tensor,
                        box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mask/conf: [B,H,W], box: [B,H,W]
    """
    mask_cm = mask.clone()
    conf_cm = conf.clone()
    mask_cm[box == 1] = mask_mix[box == 1]
    conf_cm[box == 1] = conf_mix[box == 1]
    return mask_cm, conf_cm


def ensure_cls_shape(y_cls: torch.Tensor) -> torch.Tensor:
    """
    Make y_cls -> [B,C] float for BCELoss.
    Handles [B], [B,1], [B,C].
    """
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(
    args,
    model,
    optimizer,
    loader_l,
    loader_u,
    loader_u_mix,
    device,
    total_iters,
    epoch,
    logger,
    use_amp,
    amp_dtype,
    scaler
):
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

        # to device
        x_long = x_long.to(device); x_trans = x_trans.to(device)
        m_long = m_long.to(device); m_trans = m_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        uL_w = uL_w.to(device); uL_s1 = uL_s1.to(device); uL_s2 = uL_s2.to(device)
        uT_w = uT_w.to(device); uT_s1 = uT_s1.to(device); uT_s2 = uT_s2.to(device)

        boxL1 = boxL1.to(device); boxL2 = boxL2.to(device)
        boxT1 = boxT1.to(device); boxT2 = boxT2.to(device)

        uL_wm = uL_wm.to(device); uL_s1m = uL_s1m.to(device); uL_s2m = uL_s2m.to(device)
        uT_wm = uT_wm.to(device); uT_s1m = uT_s1m.to(device); uT_s2m = uT_s2m.to(device)

        # 1) pseudo-label from weak-mix
        with torch.no_grad():
            model.eval()
            segL_wm, segT_wm, cls_wm = model(uL_wm, uT_wm)
            segL_wm = segL_wm.detach()
            segT_wm = segT_wm.detach()
            cls_wm = cls_wm.detach()

            confL_wm, maskL_wm = pseudo_from_logits(segL_wm)
            confT_wm, maskT_wm = pseudo_from_logits(segT_wm)

        # 2) CutMix on strong images (each view separately)
        uL_s1 = cutmix_apply_image(uL_s1, uL_s1m, boxL1)
        uL_s2 = cutmix_apply_image(uL_s2, uL_s2m, boxL2)
        uT_s1 = cutmix_apply_image(uT_s1, uT_s1m, boxT1)
        uT_s2 = cutmix_apply_image(uT_s2, uT_s2m, boxT2)

        model.train()

        num_l_bs = x_long.size(0)
        num_u_bs = uL_w.size(0)

        # 3) forward labeled + weak unlabeled with need_fp=True (fp consistency)
        x_long_all = torch.cat([x_long, uL_w], dim=0)
        x_trans_all = torch.cat([x_trans, uT_w], dim=0)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            (segL_all, segL_fp_all), (segT_all, segT_fp_all), (cls_all, cls_fp_all) = model(
                x_long_all, x_trans_all, need_fp=True
            )

            # split back
            segL_x, segL_u_w = segL_all.split([num_l_bs, num_u_bs], dim=0)
            segT_x, segT_u_w = segT_all.split([num_l_bs, num_u_bs], dim=0)

            cls_x, cls_u_w = cls_all.split([num_l_bs, num_u_bs], dim=0)

            segL_u_w_fp = segL_fp_all[num_l_bs:]   # [B,K,H,W]
            segT_u_w_fp = segT_fp_all[num_l_bs:]   # [B,K,H,W]

            # 4) strong forward (s1 & s2 together)
            uL_s = torch.cat([uL_s1, uL_s2], dim=0)
            uT_s = torch.cat([uT_s1, uT_s2], dim=0)
            segL_s_out, segT_s_out, cls_s_out = model(uL_s, uT_s)

            segL_s1, segL_s2 = segL_s_out.chunk(2, dim=0)
            segT_s1, segT_s2 = segT_s_out.chunk(2, dim=0)

            # 5) pseudo-label from weak (non-mix)
            segL_u_w_detach = segL_u_w.detach()
            segT_u_w_detach = segT_u_w.detach()

            confL_w, maskL_w = pseudo_from_logits(segL_u_w_detach)
            confT_w, maskT_w = pseudo_from_logits(segT_u_w_detach)

            # CutMix pseudo-labels for strong s1/s2
            maskL_cm1, confL_cm1 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL1)
            maskL_cm2, confL_cm2 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL2)

            maskT_cm1, confT_cm1 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT1)
            maskT_cm2, confT_cm2 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT2)

            # 6) losses
            # labeled seg loss: long + trans average
            loss_x_long = (criterion_seg_ce(segL_x, m_long) +
                        criterion_seg_dice(segL_x, m_long, softmax=True, ignore=torch.zeros_like(m_long))) / 2.0
            loss_x_trans = (criterion_seg_ce(segT_x, m_trans) +
                            criterion_seg_dice(segT_x, m_trans, softmax=True, ignore=torch.zeros_like(m_trans))) / 2.0
            loss_x_seg = (loss_x_long + loss_x_trans) / 2.0

            # labeled cls loss (fused cls_x already)
            loss_x_cls = criterion_cls(cls_x, y_cls.float())

            # unlabeled strong seg loss (Dice on pseudo)
            ignL1 = (confL_cm1 < args.conf_thresh).float()
            ignL2 = (confL_cm2 < args.conf_thresh).float()
            ignT1 = (confT_cm1 < args.conf_thresh).float()
            ignT2 = (confT_cm2 < args.conf_thresh).float()

            loss_uL_s = (criterion_seg_dice(segL_s1, maskL_cm1, softmax=True, ignore=ignL1) +
                        criterion_seg_dice(segL_s2, maskL_cm2, softmax=True, ignore=ignL2)) / 2.0
            loss_uT_s = (criterion_seg_dice(segT_s1, maskT_cm1, softmax=True, ignore=ignT1) +
                        criterion_seg_dice(segT_s2, maskT_cm2, softmax=True, ignore=ignT2)) / 2.0
            loss_u_s_seg = (loss_uL_s + loss_uT_s) / 2.0

            # fp loss on weak (both views)
            ignLw = (confL_w < args.conf_thresh).float()
            ignTw = (confT_w < args.conf_thresh).float()
            loss_fp_L = criterion_seg_dice(segL_u_w_fp, maskL_w, softmax=True, ignore=ignLw)
            loss_fp_T = criterion_seg_dice(segT_u_w_fp, maskT_w, softmax=True, ignore=ignTw)
            loss_u_w_fp_seg = (loss_fp_L + loss_fp_T) / 2.0

            # extra cls constraint: strong-mix pairs should match weak-mix cls_wm
            _, _, cls_s1m = model(uL_s1m, uT_s1m)
            _, _, cls_s2m = model(uL_s2m, uT_s2m)
            loss_u_s_cls = (criterion_cls_mse(torch.sigmoid(cls_s1m), torch.sigmoid(cls_wm))
                            + criterion_cls_mse(torch.sigmoid(cls_s2m), torch.sigmoid(cls_wm))) / 2.0

            # total loss
            loss = (
                loss_x_seg + loss_x_cls
                + loss_u_s_seg * 0.5
                + loss_u_w_fp_seg * 0.5
                + loss_u_s_cls * 0.1
            )

            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # poly lr
        iters = epoch * len(loader_u) + i
        lr = args.base_lr * (1 - iters / total_iters) ** 0.9
        optimizer.param_groups[0]["lr"] = lr

        # meters
        total_loss.update(loss.item())
        total_loss_x.update((loss_x_seg.item() + loss_x_cls.item()))
        total_loss_s.update(loss_u_s_seg.item())
        total_loss_fp.update(loss_u_w_fp_seg.item())
        total_loss_extra_cls.update(loss_u_s_cls.item())

        mask_ratio = (confL_w >= args.conf_thresh).sum() / confL_w.numel()
        total_mask_ratio.update(mask_ratio.item())

        if i % max(1, (len(loader_u) // 8)) == 0:
            logger.info(
                f"Iters: {i:4d} | "
                f"Total: {total_loss.avg:.3f} | "
                f"Loss_x: {total_loss_x.avg:.3f} | "
                f"Loss_s: {total_loss_s.avg:.3f} | "
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
    val_idx = 0

    for (x_long, x_trans, m_long, m_trans, y_cls) in valid_loader:
        x_long  = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long  = m_long.to(device)
        m_trans = m_trans.to(device)
        y_cls   = y_cls.to(device)

        hL, wL = x_long.shape[-2:]
        hT, wT = x_trans.shape[-2:]

        x_long_r  = F.interpolate(x_long,  (args.resize_target, args.resize_target), mode="bilinear", align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target), mode="bilinear", align_corners=False)

        segL, segT, cls_out = model(x_long_r, x_trans_r)

        cls_prob = torch.sigmoid(cls_out)
        cls_pred = (cls_prob >= 0.5).long().view(-1)
        cls_pred_list.extend(cls_pred.cpu().numpy().tolist())
        cls_gt_list.extend(y_cls.view(-1).cpu().numpy().tolist())

        segL = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
        segT = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)

        predL = torch.argmax(segL, dim=1)
        predT = torch.argmax(segT, dim=1)

        # TensorBoard visualisation
        if writer is not None:
            try:
                xL = x_long[0, 0].cpu().numpy()
                xT = x_trans[0, 0].cpu().numpy()

                def normalize_im(im):
                    mn, mx = im.min(), im.max()
                    if mx - mn < 1e-8:
                        return im - mn
                    return (im - mn) / (mx - mn)

                xL_n = normalize_im(xL)
                xT_n = normalize_im(xT)

                predL_np = predL[0].cpu().numpy()
                predT_np = predT[0].cpu().numpy()
                gtL_np   = m_long[0].cpu().numpy()
                gtT_np   = m_trans[0].cpu().numpy()

                baseL = np.stack([xL_n, xL_n, xL_n], axis=0)
                baseT = np.stack([xT_n, xT_n, xT_n], axis=0)

                def overlay(base, mask, color):
                    over  = base.copy()
                    alpha = 0.5
                    for c in range(3):
                        over[c][mask] = over[c][mask] * (1 - alpha) + color[c] * alpha
                    return over

                red   = [1.0, 0.0, 0.0]
                green = [0.0, 1.0, 0.0]

                predL_vis = overlay(overlay(baseL.copy(), predL_np == 1, red),  predL_np == 2, green)
                gtL_vis   = overlay(overlay(baseL.copy(), gtL_np   == 1, red),  gtL_np   == 2, green)
                predT_vis = overlay(overlay(baseT.copy(), predT_np == 1, red),  predT_np == 2, green)
                gtT_vis   = overlay(overlay(baseT.copy(), gtT_np   == 1, red),  gtT_np   == 2, green)

                concatL = np.concatenate([baseL, predL_vis, gtL_vis], axis=2)
                concatT = np.concatenate([baseT, predT_vis, gtT_vis], axis=2)

                step = epoch if epoch is not None and epoch >= 0 else 0
                writer.add_image(f"Val/vis/long/{val_idx}",  torch.from_numpy(concatL).float(), global_step=step)
                writer.add_image(f"Val/vis/trans/{val_idx}", torch.from_numpy(concatT).float(), global_step=step)
            except Exception as e:
                logger.warning(f"Failed to write val visualization: {e}")

        val_idx += 1

        for cls in [1, 2]:
            interL = ((predL == cls) & (m_long  == cls)).sum().item()
            unionL = (predL == cls).sum().item() + (m_long  == cls).sum().item()
            dice_long[cls]  += 2.0 * interL / (unionL + 1e-8)

            interT = ((predT == cls) & (m_trans == cls)).sum().item()
            unionT = (predT == cls).sum().item() + (m_trans == cls).sum().item()
            dice_trans[cls] += 2.0 * interT / (unionT + 1e-8)

            predL_np = (predL[0] == cls).cpu().numpy()
            gtL_np   = (m_long[0]  == cls).cpu().numpy()
            nsd_long[cls]  += compute_nsd(predL_np, gtL_np,  tolerance=3.0)

            predT_np = (predT[0] == cls).cpu().numpy()
            gtT_np   = (m_trans[0] == cls).cpu().numpy()
            nsd_trans[cls] += compute_nsd(predT_np, gtT_np, tolerance=3.0)

    idx_to_name = {1: "Plaque", 2: "Vessel"}
    for cls in [1, 2]:
        dice_long[cls]  /= max(1, num_batches)
        dice_trans[cls] /= max(1, num_batches)
        logger.info(f"[Dice] {idx_to_name[cls]} | Long Dice: {dice_long[cls]:.2f} | Trans Dice: {dice_trans[cls]:.2f}")

        nsd_long[cls]  /= max(1, num_batches)
        nsd_trans[cls] /= max(1, num_batches)
        logger.info(f"[NSD] {idx_to_name[cls]} | Long NSD: {nsd_long[cls]:.2f} | Trans NSD: {nsd_trans[cls]:.2f}")

    mean_dice = (dice_long[1] + dice_long[2] + dice_trans[1] + dice_trans[2]) / 4.0
    mean_NSD  = (nsd_long[1]  + nsd_long[2]  + nsd_trans[1]  + nsd_trans[2]) / 4.0
    logger.info(f"[Dice] Mean Foreground Dice: {mean_dice:.3f}")
    logger.info(f"[NSD] Mean Foreground NSD: {mean_NSD:.3f}")

    cls_gt   = np.array(cls_gt_list)
    cls_pred = np.array(cls_pred_list)
    f1       = f1_score(cls_gt, cls_pred)
    logger.info(f"[Cls] F1 Score: {f1:.4f}")

    cm     = confusion_matrix(cls_gt, cls_pred)
    labels = list(range(cm.shape[0]))

    def _format_confusion_matrix(cm_array, lbls):
        header = "\t" + "\t".join([f"Pred:{l}" for l in lbls])
        lines  = [header]
        for i, l in enumerate(lbls):
            counts    = "\t".join(str(int(x)) for x in cm_array[i])
            row_total = cm_array[i].sum()
            percents  = (
                "\t".join(f"{(cm_array[i, j] / row_total * 100):.1f}%" for j in range(cm_array.shape[1]))
                if row_total > 0 else
                "\t".join("0.0%" for _ in range(cm_array.shape[1]))
            )
            lines.append(f"True:{l}\t{counts}\t| {percents}")
        return "\n".join(lines)

    logger.info("Confusion Matrix (rows=true labels, cols=pred labels):\n" + _format_confusion_matrix(cm, labels))

    return {
        "dice_long_vessel":  dice_long[2],
        "dice_long_plaque":  dice_long[1],
        "dice_trans_vessel": dice_trans[2],
        "dice_trans_plaque": dice_trans[1],

        "nsd_long_vessel":  nsd_long[2],
        "nsd_long_plaque":  nsd_long[1],
        "nsd_trans_vessel": nsd_trans[2],
        "nsd_trans_plaque": nsd_trans[1],

        "cls_score": f1,

        "seg_socre_long_vessel":  (dice_long[2]  + nsd_long[2])  / 2,
        "seg_socre_long_plaque":  (dice_long[1]  + nsd_long[1])  / 2,
        "seg_socre_trans_vessel": (dice_trans[2] + nsd_trans[2]) / 2,
        "seg_socre_trans_plaque": (dice_trans[1] + nsd_trans[1]) / 2,

        "seg_score": (
            (dice_long[2]  + nsd_long[2])  / 2 * 0.4 +
            (dice_long[1]  + nsd_long[1])  / 2 * 0.6 +
            (dice_trans[2] + nsd_trans[2]) / 2 * 0.4 +
            (dice_trans[1] + nsd_trans[1]) / 2 * 0.6
        ) / 2,

        # total_score capped ~0.8 locally (time component not evaluated here)
        "total_score": (
            f1 * 0.4 +
            (dice_long[2]  + nsd_long[2])  / 2 * 0.4 * 0.2 +
            (dice_long[1]  + nsd_long[1])  / 2 * 0.6 * 0.2 +
            (dice_trans[2] + nsd_trans[2]) / 2 * 0.4 * 0.2 +
            (dice_trans[1] + nsd_trans[1]) / 2 * 0.6 * 0.2
        ),
    }


def build_logger(save_path: str):
    logger = logging.getLogger("UniMatch TwoView Training")
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


def get_model(args):
    if args.model == "Echocare":
        model = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.echo_care_ckpt,
        )
    elif args.model == "UNet":
        model = UNetTwoView(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    else:
        raise ValueError(f"Unknown model choice: {args.model}")
    return model


if __name__ == "__main__":
    main()
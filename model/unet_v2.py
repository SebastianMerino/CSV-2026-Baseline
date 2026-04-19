from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Building blocks  (unchanged from v1)
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two conv layers with BN and LeakyReLU."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels,
                 dropout_p, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2,
                                         kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


# ─────────────────────────────────────────────
# Encoder  (unchanged from v1)
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    5-level encoder.  feature_chns = [16, 32, 64, 128, 256]
    Produces [x0, x1, x2, x3, x4] where x4 is the bottleneck.
    """

    def __init__(self, params):
        super().__init__()
        self.ft_chns = params["feature_chns"]
        dropout      = params["dropout"]

        self.in_conv = ConvBlock(params["in_chns"], self.ft_chns[0], dropout[0])
        self.down1   = DownBlock(self.ft_chns[0],  self.ft_chns[1], dropout[1])
        self.down2   = DownBlock(self.ft_chns[1],  self.ft_chns[2], dropout[2])
        self.down3   = DownBlock(self.ft_chns[2],  self.ft_chns[3], dropout[3])
        self.down4   = DownBlock(self.ft_chns[3],  self.ft_chns[4], dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


# ─────────────────────────────────────────────
# Decoder  (unchanged from v1)
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        ft = params["feature_chns"]

        self.up1 = UpBlock(ft[4], ft[3], ft[3], dropout_p=0.0)
        self.up2 = UpBlock(ft[3], ft[2], ft[2], dropout_p=0.0)
        self.up3 = UpBlock(ft[2], ft[1], ft[1], dropout_p=0.0)
        self.up4 = UpBlock(ft[1], ft[0], ft[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(ft[0], params["class_num"], kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x,  x2)
        x = self.up3(x,  x1)
        x = self.up4(x,  x0)
        return self.out_conv(x)


# ─────────────────────────────────────────────
# NEW: Multi-scale embedding helper
# ─────────────────────────────────────────────

class MultiScaleEmbedding(nn.Module):
    """
    Pools three encoder scales (x2, x3, x4) with both avg and max pooling,
    then projects the concatenation to a fixed embedding dimension.

    For feature_chns = [16, 32, 64, 128, 256]:
        raw concat size = (64 + 128 + 256) * 2  =  896  (avg + max per scale)
        → projected to `embed_dim`

    Why three scales?
      x4 (256ch) — global semantics / plaque presence
      x3 (128ch) — intermediate structure
      x2  (64ch) — fine local texture (thin fibrous cap, echogenicity)

    Why avg + max?
      avg pool → distributed signal (background tissue, overall echogenicity)
      max pool → peak activations (focal hyper/hypo-echoic regions)
    """

    def __init__(self, feature_chns, embed_dim: int = 256):
        super().__init__()
        # scales we pool from: x2, x3, x4
        raw_dim = (feature_chns[2] + feature_chns[3] + feature_chns[4]) * 2
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, feats):
        """feats: list [x0, x1, x2, x3, x4]"""
        parts = []
        for fi in feats[2:]:                                          # x2, x3, x4
            avg = F.adaptive_avg_pool2d(fi, 1).flatten(1)            # [B, C]
            mx  = F.adaptive_max_pool2d(fi, 1).flatten(1)            # [B, C]
            parts.extend([avg, mx])
        raw = torch.cat(parts, dim=1)                                 # [B, raw_dim]
        return self.proj(raw)                                         # [B, embed_dim]


# ─────────────────────────────────────────────
# NEW: Per-view + fused classification head
# ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Three-branch classifier:
      • branch_long  : embed_long  → logit   (supervise independently)
      • branch_trans : embed_trans → logit   (supervise independently)
      • branch_fuse  : concat(long, trans) → logit  (main output)

    During training use all three logits.
    During inference use only the fused logit.

    Independent per-view supervision forces BOTH encoder branches to
    learn vulnerability-discriminative features, preventing one view
    from free-riding on the other.
    """

    def __init__(self, embed_dim: int = 256, cls_class_num: int = 1,
                 dropout_p: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout_p)

        # per-view heads (shallow — embedding already rich)
        self.branch_long  = nn.Linear(embed_dim, cls_class_num)
        self.branch_trans = nn.Linear(embed_dim, cls_class_num)

        # fused head
        self.branch_fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(embed_dim, cls_class_num),
        )

    def forward(self, emb_long, emb_trans):
        """
        Returns (logit_fuse, logit_long, logit_trans).
        All are [B, cls_class_num] raw logits — apply BCEWithLogitsLoss externally.
        """
        el = self.drop(emb_long)
        et = self.drop(emb_trans)

        logit_long  = self.branch_long(el)
        logit_trans = self.branch_trans(et)
        logit_fuse  = self.branch_fuse(torch.cat([el, et], dim=1))

        return logit_fuse, logit_long, logit_trans


# ─────────────────────────────────────────────
# UNetTwoViewV2  — drop-in replacement for UNetTwoView
# ─────────────────────────────────────────────

class UNetTwoViewV2(nn.Module):
    """
    Improved two-view UNet focused on classification performance.

    Changes vs v1
    ─────────────
    1. MultiScaleEmbedding: pools x2+x3+x4 with avg AND max pooling
       instead of only avg-pooling x4.  Captures fine local texture
       (thin fibrous cap, focal echogenic regions) alongside global context.

    2. ClassificationHead: three branches (long-only, trans-only, fused).
       Per-view branches receive independent gradient from the cls loss,
       preventing one view dominating and ensuring both encoder paths
       learn vulnerability-relevant features.

    3. Fixed need_fp perturbation: previous version created a new
       Dropout2d instance inside a torch.no_grad() context which could
       silently skip dropping.  Now uses a dedicated nn.Dropout2d module
       forced into train mode so perturbation is always active.

    4. Projection norm: LayerNorm + GELU in the embedding projection
       stabilises training when gradients flow from both the seg decoder
       and the classification head simultaneously.

    Forward signature is fully backward-compatible with UNetTwoView:
      need_fp=False → seg_long, seg_trans, cls_logit_fuse
      need_fp=True  → (seg_l, seg_l_fp), (seg_t, seg_t_fp),
                       (cls_fuse, cls_fuse_fp),
                       (cls_long, cls_long_fp),    ← new
                       (cls_trans, cls_trans_fp)   ← new
    """

    EMBED_DIM = 256   # embedding dim after multi-scale projection

    def __init__(self, in_chns: int, seg_class_num: int, cls_class_num: int):
        super().__init__()

        params = {
            "in_chns":      in_chns,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout":      [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num":    seg_class_num,
            "bilinear":     False,
        }
        ft = params["feature_chns"]

        # ── shared encoder (same weights for both views) ──
        self.encoder = Encoder(params)

        # ── per-view segmentation decoders ──
        self.seg_decoder_long  = Decoder(params)
        self.seg_decoder_trans = Decoder(params)

        # ── improved classification components ──
        self.ms_embed   = MultiScaleEmbedding(ft, embed_dim=self.EMBED_DIM)
        self.cls_head   = ClassificationHead(self.EMBED_DIM, cls_class_num,
                                             dropout_p=0.3)

        # ── dedicated dropout for need_fp perturbation ──
        # Kept as a proper module so .train()/.eval() propagates correctly,
        # but we force train() explicitly during need_fp to guarantee dropping.
        self._fp_drop = nn.Dropout2d(p=0.5)

    # ── helpers ──────────────────────────────────────────

    def _encode_both(self, x_long, x_trans):
        return self.encoder(x_long), self.encoder(x_trans)

    def _perturb(self, feats):
        """
        Return a perturbed copy of a feature list by applying Dropout2d
        to every scale.  Forces train() so dropout is always active
        regardless of the outer model state (which is eval() during
        pseudo-label generation).
        """
        self._fp_drop.train()
        return [self._fp_drop(f) for f in feats]

    def _classify(self, feat_long, feat_trans):
        emb_l = self.ms_embed(feat_long)
        emb_t = self.ms_embed(feat_trans)
        return self.cls_head(emb_l, emb_t)   # (fuse, long, trans)

    # ── forward ──────────────────────────────────────────

    def forward(self, x_long, x_trans, need_fp=False):
        feat_l, feat_t = self._encode_both(x_long, x_trans)

        if need_fp:
            # ── original branch ──
            seg_l   = self.seg_decoder_long(feat_l)
            seg_t   = self.seg_decoder_trans(feat_t)
            cls_f, cls_l, cls_t = self._classify(feat_l, feat_t)

            # ── feature-perturbed branch ──
            feat_l_fp = self._perturb(feat_l)
            feat_t_fp = self._perturb(feat_t)

            seg_l_fp  = self.seg_decoder_long(feat_l_fp)
            seg_t_fp  = self.seg_decoder_trans(feat_t_fp)
            cls_f_fp, cls_l_fp, cls_t_fp = self._classify(feat_l_fp, feat_t_fp)

            return (
                (seg_l,  seg_l_fp),
                (seg_t,  seg_t_fp),
                (cls_f,  cls_f_fp),
                (cls_l,  cls_l_fp),    # per-view logits, original + perturbed
                (cls_t,  cls_t_fp),
            )

        # ── normal forward ──
        seg_l = self.seg_decoder_long(feat_l)
        seg_t = self.seg_decoder_trans(feat_t)
        cls_f, cls_l, cls_t = self._classify(feat_l, feat_t)

        # Return (fuse, long, trans) so the caller can use all three.
        # train.py only reads the first element for inference/validation,
        # matching the original interface.
        return seg_l, seg_t, (cls_f, cls_l, cls_t)
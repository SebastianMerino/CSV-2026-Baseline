#!/usr/bin/env python3
"""
post_seg_classifier.py

Post-segmentation classifier that uses:
1. Segmentation probability maps (plaque morphology, vessel structure)
2. Raw images (texture information)
3. Handcrafted features from segmentation (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFeatureExtractor(nn.Module):
    """
    CNN to extract features from segmentation probability maps.
    """
    
    def __init__(self, seg_classes=3, out_dim=128):
        super().__init__()
        
        # Input: [B, seg_classes*2, H, W] (long + trans concatenated)
        self.conv = nn.Sequential(
            nn.Conv2d(seg_classes * 2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, seg_prob_long, seg_prob_trans):
        """
        Args:
            seg_prob_long: [B, seg_classes, H, W] softmax probabilities
            seg_prob_trans: [B, seg_classes, H, W]
        """
        # Resize to common size
        target_size = (64, 64)
        seg_L = F.interpolate(seg_prob_long, target_size, mode='bilinear', align_corners=False)
        seg_T = F.interpolate(seg_prob_trans, target_size, mode='bilinear', align_corners=False)
        
        # Concatenate views
        combined = torch.cat([seg_L, seg_T], dim=1)  # [B, 6, 64, 64]
        
        feat = self.conv(combined)
        return self.fc(feat)


class ImageFeatureExtractor(nn.Module):
    """
    Small CNN to extract texture features from raw images.
    """
    
    def __init__(self, in_channels=2, out_dim=64):
        super().__init__()
        
        # Input: [B, 2, H, W] (long + trans concatenated)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, img_long, img_trans):
        """
        Args:
            img_long: [B, 1, H, W]
            img_trans: [B, 1, H, W]
        """
        target_size = (64, 64)
        img_L = F.interpolate(img_long, target_size, mode='bilinear', align_corners=False)
        img_T = F.interpolate(img_trans, target_size, mode='bilinear', align_corners=False)
        
        combined = torch.cat([img_L, img_T], dim=1)  # [B, 2, 64, 64]
        
        feat = self.conv(combined)
        return self.fc(feat)


class HandcraftedSegFeatures(nn.Module):
    """
    Extract interpretable handcrafted features from segmentation masks.
    """
    
    def __init__(self, seg_classes=3):
        super().__init__()
        self.seg_classes = seg_classes
        # Features per view: class_ratios(3) + plaque_features(4) + vessel_features(2) = 9
        # Total: 9 * 2 views = 18
        self.out_dim = 18
    
    def forward(self, seg_pred_long, seg_pred_trans):
        """
        Args:
            seg_pred_long: [B, H, W] predicted class indices (argmax)
            seg_pred_trans: [B, H, W]
        
        Returns:
            features: [B, out_dim]
        """
        B = seg_pred_long.shape[0]
        device = seg_pred_long.device
        
        all_features = []
        
        for b in range(B):
            features = []
            
            for pred in [seg_pred_long[b], seg_pred_trans[b]]:
                total_pixels = pred.numel()
                
                # Class ratios
                for c in range(self.seg_classes):
                    ratio = (pred == c).float().sum() / total_pixels
                    features.append(ratio)
                
                # Plaque features (assuming class 1 = plaque)
                plaque_mask = (pred == 1)
                plaque_pixels = plaque_mask.float().sum()
                
                if plaque_pixels > 10:  # Minimum threshold
                    # Plaque area ratio
                    plaque_ratio = plaque_pixels / total_pixels
                    
                    # Bounding box features
                    coords = torch.nonzero(plaque_mask)
                    h_min, h_max = coords[:, 0].min(), coords[:, 0].max()
                    w_min, w_max = coords[:, 1].min(), coords[:, 1].max()
                    
                    h_range = (h_max - h_min + 1).float()
                    w_range = (w_max - w_min + 1).float()
                    
                    # Aspect ratio
                    aspect = h_range / (w_range + 1e-6)
                    
                    # Compactness (actual area / bounding box area)
                    bbox_area = h_range * w_range
                    compactness = plaque_pixels / (bbox_area + 1e-6)
                    
                    # Eccentricity approximation
                    eccentricity = torch.abs(h_range - w_range) / (torch.max(h_range, w_range) + 1e-6)
                    
                    features.extend([plaque_ratio, aspect, compactness, eccentricity])
                else:
                    features.extend([torch.tensor(0.0, device=device)] * 4)
                
                # Vessel features (assuming class 2 = vessel)
                vessel_mask = (pred == 2)
                vessel_pixels = vessel_mask.float().sum()
                
                # Vessel area ratio
                vessel_ratio = vessel_pixels / total_pixels
                
                # Stenosis indicator (plaque / vessel ratio)
                if vessel_pixels > 10:
                    stenosis = plaque_pixels / (vessel_pixels + 1e-6)
                else:
                    stenosis = torch.tensor(0.0, device=device)
                
                features.extend([vessel_ratio, stenosis])
            
            # Stack and ensure tensor
            feat_tensor = torch.stack([
                f if torch.is_tensor(f) else torch.tensor(f, device=device) 
                for f in features
            ])
            all_features.append(feat_tensor)
        
        return torch.stack(all_features)  # [B, out_dim]


class PostSegClassifier(nn.Module):
    """
    Complete post-segmentation classifier.
    
    Combines:
    1. Segmentation CNN features (morphology)
    2. Image CNN features (texture)
    3. Handcrafted segmentation features (interpretable)
    """
    
    def __init__(self, seg_classes=3, cls_classes=1, 
                 use_seg_cnn=True, use_img_cnn=True, use_handcrafted=True,
                 dropout=0.4):
        super().__init__()
        
        self.use_seg_cnn = use_seg_cnn
        self.use_img_cnn = use_img_cnn
        self.use_handcrafted = use_handcrafted
        
        # Feature extractors
        self.seg_feat_dim = 0
        self.img_feat_dim = 0
        self.hc_feat_dim = 0
        
        if use_seg_cnn:
            self.seg_extractor = SegFeatureExtractor(seg_classes=seg_classes, out_dim=128)
            self.seg_feat_dim = 128
        
        if use_img_cnn:
            self.img_extractor = ImageFeatureExtractor(in_channels=2, out_dim=64)
            self.img_feat_dim = 64
        
        if use_handcrafted:
            self.hc_extractor = HandcraftedSegFeatures(seg_classes=seg_classes)
            self.hc_feat_dim = self.hc_extractor.out_dim
        
        total_feat_dim = self.seg_feat_dim + self.img_feat_dim + self.hc_feat_dim
        
        if total_feat_dim == 0:
            raise ValueError("At least one feature extractor must be enabled!")
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(64, cls_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, img_long, img_trans, seg_logits_long, seg_logits_trans):
        """
        Args:
            img_long: [B, 1, H, W] raw image
            img_trans: [B, 1, H, W] raw image
            seg_logits_long: [B, C, H, W] segmentation logits
            seg_logits_trans: [B, C, H, W] segmentation logits
        
        Returns:
            cls_logits: [B, cls_classes]
        """
        features = []
        
        # Segmentation probabilities
        seg_prob_L = torch.softmax(seg_logits_long, dim=1)
        seg_prob_T = torch.softmax(seg_logits_trans, dim=1)
        
        # Segmentation CNN features
        if self.use_seg_cnn:
            seg_feat = self.seg_extractor(seg_prob_L, seg_prob_T)
            features.append(seg_feat)
        
        # Image CNN features
        if self.use_img_cnn:
            img_feat = self.img_extractor(img_long, img_trans)
            features.append(img_feat)
        
        # Handcrafted features
        if self.use_handcrafted:
            seg_pred_L = torch.argmax(seg_logits_long, dim=1)  # [B, H, W]
            seg_pred_T = torch.argmax(seg_logits_trans, dim=1)
            hc_feat = self.hc_extractor(seg_pred_L, seg_pred_T)
            features.append(hc_feat)
        
        # Concatenate and classify
        combined = torch.cat(features, dim=1)
        return self.classifier(combined)


class PostSegClassifierLight(nn.Module):
    """
    Lightweight version using only segmentation features.
    """
    
    def __init__(self, seg_classes=3, cls_classes=1, dropout=0.3):
        super().__init__()
        
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(seg_classes * 2, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, cls_classes),
        )
    
    def forward(self, img_long, img_trans, seg_logits_long, seg_logits_trans):
        seg_prob_L = torch.softmax(seg_logits_long, dim=1)
        seg_prob_T = torch.softmax(seg_logits_trans, dim=1)
        
        target_size = (64, 64)
        seg_L = F.interpolate(seg_prob_L, target_size, mode='bilinear', align_corners=False)
        seg_T = F.interpolate(seg_prob_T, target_size, mode='bilinear', align_corners=False)
        
        combined = torch.cat([seg_L, seg_T], dim=1)
        feat = self.seg_encoder(combined)
        return self.classifier(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_post_classifier(variant="full", seg_classes=3, cls_classes=1, dropout=0.4):
    """
    Factory function to create post-segmentation classifier.
    
    Args:
        variant: "full", "seg_only", "seg_img", "light"
        seg_classes: Number of segmentation classes
        cls_classes: Number of classification classes
        dropout: Dropout rate
    """
    if variant == "full":
        return PostSegClassifier(
            seg_classes=seg_classes,
            cls_classes=cls_classes,
            use_seg_cnn=True,
            use_img_cnn=True,
            use_handcrafted=True,
            dropout=dropout,
        )
    elif variant == "seg_only":
        return PostSegClassifier(
            seg_classes=seg_classes,
            cls_classes=cls_classes,
            use_seg_cnn=True,
            use_img_cnn=False,
            use_handcrafted=False,
            dropout=dropout,
        )
    elif variant == "seg_img":
        return PostSegClassifier(
            seg_classes=seg_classes,
            cls_classes=cls_classes,
            use_seg_cnn=True,
            use_img_cnn=True,
            use_handcrafted=False,
            dropout=dropout,
        )
    elif variant == "light":
        return PostSegClassifierLight(
            seg_classes=seg_classes,
            cls_classes=cls_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
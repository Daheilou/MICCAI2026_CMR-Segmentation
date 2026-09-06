# metrics.py
# Segmentation metrics: per-class Dice, IoU, 95% Hausdorff Distance (HD95).
# Also provides DiceLoss as a differentiable training criterion.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List


# ═══════════════════════════════════════════════════════════
#  Training loss
# ═══════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Differentiable soft Dice loss for multi-class segmentation.

    Converts logits to per-class probabilities via softmax, then computes
    the mean soft Dice loss over all foreground classes (class 0 = background
    is excluded by default).

    Args:
        num_classes      : total number of classes (including background)
        smooth           : Laplace smoothing constant
        ignore_background: if True, class-0 loss is excluded from the mean
    """
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-6,
        ignore_background: bool = True,
    ):
        super().__init__()
        self.num_classes       = num_classes
        self.smooth            = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, C, H, W)  raw network output
            target : (B, H, W)     integer class labels in [0, C)

        Returns:
            Scalar Dice loss (1 - mean_dice).
        """
        probs = torch.softmax(logits, dim=1)          # (B, C, H, W)

        # One-hot encode target → (B, C, H, W)
        B, C, H, W = probs.shape
        target_oh  = torch.zeros_like(probs)
        target_oh.scatter_(1, target.unsqueeze(1), 1.0)

        start_cls = 1 if self.ignore_background else 0
        dice_vals = []
        for c in range(start_cls, C):
            p = probs[:, c]            # (B, H, W)
            t = target_oh[:, c]        # (B, H, W)
            intersection = (p * t).sum()
            union        = p.sum() + t.sum()
            dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_vals.append(dice)

        if not dice_vals:
            return torch.tensor(0.0, device=logits.device)
        return 1.0 - torch.stack(dice_vals).mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of CrossEntropyLoss and DiceLoss.

    Total loss = ce_weight * CE + dice_weight * Dice

    Args:
        num_classes : number of output classes (per dataset; set dynamically)
        ce_weight   : weight for CE term (default 0.5)
        dice_weight : weight for Dice term (default 0.5)
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce          = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Args:
            logits      : (B, C, H, W)
            target      : (B, H, W) integer labels
            num_classes : C (used to instantiate DiceLoss on-the-fly;
                          cached per C to avoid repeated object creation)
        Returns:
            Scalar combined loss.
        """
        # Lazy-cache DiceLoss per num_classes to avoid re-creating every step
        if not hasattr(self, '_dice_cache'):
            self._dice_cache = {}
        if num_classes not in self._dice_cache:
            self._dice_cache[num_classes] = DiceLoss(num_classes).to(logits.device)

        dice_loss_fn = self._dice_cache[num_classes]
        ce_loss   = self.ce(logits, target)
        dice_loss = dice_loss_fn(logits, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class OneVsRestCombinedLoss(nn.Module):
    """
    Loss for independent foreground binary heads.

    Inputs:
        fg_logits : (B, K, H, W), K = num_classes - 1
        fg_target : (B, K, H, W), binary labels per foreground class

    Total = bce_weight * BCE/Focal + dice_weight * weighted binary soft dice
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.smooth = smooth

    def _binary_dice_loss(
        self,
        fg_logits: torch.Tensor,
        fg_target: torch.Tensor,
        class_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(fg_logits)
        B, K, _, _ = probs.shape
        if K == 0:
            return torch.tensor(0.0, device=probs.device)

        dice_vals = []
        for class_idx in range(K):
            p = probs[:, class_idx]
            t = fg_target[:, class_idx]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            if class_weights is not None:
                dice = dice * class_weights[class_idx]
            dice_vals.append(dice)

        if class_weights is not None:
            norm = class_weights.sum().clamp_min(1e-8)
            return 1.0 - (torch.stack(dice_vals).sum() / norm)

        return 1.0 - torch.stack(dice_vals).mean()

    def _bce_or_focal(
        self,
        fg_logits: torch.Tensor,
        fg_target: torch.Tensor,
        pos_weight: torch.Tensor = None,
        class_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        loss_map = F.binary_cross_entropy_with_logits(
            fg_logits,
            fg_target,
            reduction='none',
        )
        if pos_weight is not None:
            pos_w = pos_weight.view(1, -1, 1, 1)
            pos_mask = (fg_target > 0.5).float()
            weight_map = 1.0 + pos_mask * (pos_w - 1.0)
            loss_map = loss_map * weight_map

        if self.focal_weight > 0:
            probs = torch.sigmoid(fg_logits)
            pt = torch.where(fg_target > 0.5, probs, 1.0 - probs)
            focal = (1.0 - pt).pow(self.focal_gamma)
            loss_map = (1.0 - self.focal_weight) * loss_map + self.focal_weight * (focal * loss_map)

        if class_weights is not None:
            class_w = class_weights.view(1, -1, 1, 1)
            loss_map = loss_map * class_w
            return loss_map.sum() / class_w.sum().clamp_min(1e-8) / (fg_logits.shape[0] * fg_logits.shape[2] * fg_logits.shape[3])

        return loss_map.mean()

    def forward(
        self,
        fg_logits: torch.Tensor,
        fg_target: torch.Tensor,
        pos_weight: torch.Tensor = None,
        class_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        bce_loss = self._bce_or_focal(fg_logits, fg_target, pos_weight=pos_weight, class_weights=class_weights)
        dice_loss = self._binary_dice_loss(fg_logits, fg_target, class_weights=class_weights)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ═══════════════════════════════════════════════════════════
#  Evaluation metrics
# ═══════════════════════════════════════════════════════════

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> List[float]:
    """
    Per-class Dice coefficient.

    Args:
        pred        : (B, H, W) integer class predictions
        target      : (B, H, W) integer ground-truth labels
        num_classes : total number of classes (including background)
        smooth      : Laplace smoothing

    Returns:
        List[float] of length `num_classes`.
    """
    dice_per_class = []
    for c in range(num_classes):
        pred_c        = (pred   == c).float()
        target_c      = (target == c).float()
        intersection  = (pred_c * target_c).sum()
        union         = pred_c.sum() + target_c.sum()
        dice          = (2.0 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice.item())
    return dice_per_class


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> List[float]:
    """
    Per-class Intersection-over-Union (Jaccard index).

    Args:
        pred        : (B, H, W) integer class predictions
        target      : (B, H, W) integer ground-truth labels
        num_classes : total number of classes
        smooth      : Laplace smoothing

    Returns:
        List[float] of length `num_classes`.
    """
    iou_per_class = []
    for c in range(num_classes):
        pred_c        = (pred   == c).float()
        target_c      = (target == c).float()
        intersection  = (pred_c * target_c).sum()
        union         = (pred_c + target_c).clamp(0, 1).sum()
        iou           = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou.item())
    return iou_per_class


def hausdorff_distance_95(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    voxel_spacing: float = 1.0,
) -> List[float]:
    """
    Per-class 95th-percentile Hausdorff Distance (HD95).

    Computes symmetric HD95 for each class by measuring distances from the
    predicted surface to the ground-truth surface (and vice-versa) using
    Euclidean Distance Transform (scipy).

    Args:
        pred          : (B, H, W) integer class predictions  (CPU or GPU)
        target        : (B, H, W) integer ground-truth labels
        num_classes   : total number of classes
        voxel_spacing : isotropic pixel/voxel size in mm (default 1.0)

    Returns:
        List[float] of length `num_classes`.
        Classes where both pred and target are empty return 0.0.
        Classes where only one side is empty return a large sentinel (999.0).
    """
    # Move to CPU numpy once
    pred_np   = pred.cpu().numpy().astype(np.uint8)    # (B, H, W)
    target_np = target.cpu().numpy().astype(np.uint8)  # (B, H, W)

    hd95_per_class = []
    for c in range(num_classes):
        p_bin = (pred_np   == c)   # bool (B, H, W)
        t_bin = (target_np == c)   # bool (B, H, W)

        # Flatten batch dimension → single 2D boolean mask per class
        # We process each image separately and average
        batch_hd95 = []
        for b_idx in range(pred_np.shape[0]):
            pb = p_bin[b_idx]   # (H, W)
            tb = t_bin[b_idx]   # (H, W)

            if not pb.any() and not tb.any():
                # Both empty → perfect agreement
                batch_hd95.append(0.0)
                continue
            if not pb.any() or not tb.any():
                # One-sided miss → use a large penalty
                batch_hd95.append(999.0)
                continue

            # Distance from every pixel to nearest foreground pixel in each mask
            # EDT is computed on the *complement* (background→0, foreground→0)
            dist_pred2gt = distance_transform_edt(~tb) * voxel_spacing
            dist_gt2pred = distance_transform_edt(~pb) * voxel_spacing

            # Surface distances: EDT value at the *other* mask's surface
            surf_pred = dist_pred2gt[pb]   # distances from pred surface to gt
            surf_gt   = dist_gt2pred[tb]   # distances from gt surface to pred

            all_dists = np.concatenate([surf_pred, surf_gt])
            hd95_val  = float(np.percentile(all_dists, 95))
            batch_hd95.append(hd95_val)

        hd95_per_class.append(float(np.mean(batch_hd95)))

    return hd95_per_class


def mean_dice(dice_list: List[float], ignore_background: bool = True) -> float:
    """Mean Dice, optionally excluding class-0 (background)."""
    scores = dice_list[1:] if ignore_background and len(dice_list) > 1 else dice_list
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_iou(iou_list: List[float], ignore_background: bool = True) -> float:
    """Mean IoU, optionally excluding class-0 (background)."""
    scores = iou_list[1:] if ignore_background and len(iou_list) > 1 else iou_list
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_hd95(hd95_list: List[float], ignore_background: bool = True) -> float:
    """Mean HD95, optionally excluding class-0 (background). Lower is better."""
    scores = hd95_list[1:] if ignore_background and len(hd95_list) > 1 else hd95_list
    valid  = [v for v in scores if v < 999.0]   # exclude sentinel values
    return float(sum(valid) / len(valid)) if valid else float('inf')

"""
DETR set prediction losses: classification, box regression, mask, and DN denoising.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou

DEBUG = False


def dice_loss(pred, target, num_objects):
    """
    Dice loss for binary masks.
    Args:
        pred:   [N, H, W] logits (sigmoid applied internally)
        target: [N, H, W] binary float
        num_objects: normalisation denominator
    Returns:
        scalar loss
    """
    pred = pred.sigmoid().flatten(1)   # [N, HW]
    target = target.flatten(1)         # [N, HW]

    numerator = 2 * (pred * target).sum(1)
    denominator = pred.sum(1) + target.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / max(num_objects, 1)


class SetCriterion(nn.Module):
    """
    DETR loss that combines:
      - classification (CE with down-weighted background)
      - bounding box (L1 + GIoU) on Hungarian-matched pairs
      - optional mask (BCE + Dice) on matched pairs
      - optional DN denoising losses (direct supervision)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef   # background class weight
        self.register_buffer("empty_weight", empty_weight)

    # ------------------------------------------------------------------
    # Matching losses
    # ------------------------------------------------------------------

    def loss_labels(self, pred_logits, targets, indices, num_objects):
        """Cross-entropy classification loss."""
        # Default: all queries predict background
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=pred_logits.device,
        )
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][gt_idx]

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),   # [B, C+1, N]
            target_classes,                # [B, N]
            weight=self.empty_weight,
        )
        return loss_ce

    def loss_boxes(self, pred_boxes, targets, indices, num_objects):
        """L1 + GIoU on matched boxes."""
        src_list, tgt_list = [], []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                src_list.append(pred_boxes[b, pred_idx])
                tgt_list.append(targets[b]["boxes"][gt_idx])

        if not src_list:
            z = torch.tensor(0.0, device=pred_boxes.device)
            return z, z

        src = torch.cat(src_list, dim=0)
        tgt = torch.cat(tgt_list, dim=0)

        loss_bbox = F.l1_loss(src, tgt, reduction="sum") / max(num_objects, 1)

        src_xyxy = box_cxcywh_to_xyxy(src)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt)
        giou = generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - giou.diag()).sum() / max(num_objects, 1)

        return loss_bbox, loss_giou

    def loss_masks(self, pred_masks, targets, indices, num_objects):
        """BCE + Dice on matched masks."""
        src_list, tgt_list = [], []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0 or "masks" not in targets[b]:
                continue
            src_list.append(pred_masks[b, pred_idx])            # [K, H, W]
            gt_m = targets[b]["masks"][gt_idx].float()          # [K, H_gt, W_gt]
            if gt_m.shape[-2:] != pred_masks.shape[-2:]:
                gt_m = F.interpolate(
                    gt_m.unsqueeze(1),
                    size=pred_masks.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            tgt_list.append(gt_m)

        if not src_list:
            z = torch.tensor(0.0, device=pred_masks.device)
            return z, z

        src = torch.cat(src_list, dim=0)   # [K_total, H, W]
        tgt = torch.cat(tgt_list, dim=0)

        loss_mask = F.binary_cross_entropy_with_logits(src, tgt, reduction="mean")
        loss_dice = dice_loss(src, tgt, num_objects)
        return loss_mask, loss_dice

    # ------------------------------------------------------------------
    # DN denoising loss
    # ------------------------------------------------------------------

    def loss_dn(self, dn_pred_logits, dn_pred_boxes, dn_meta):
        """Direct supervision on denoising queries (no Hungarian matching)."""
        if dn_meta is None:
            z = torch.tensor(0.0, device=dn_pred_logits.device)
            return z, z, z

        valid = dn_meta["valid_mask"].bool()          # [B, dn_len]
        gt_labels = dn_meta["gt_labels"]              # [B, dn_len]
        gt_boxes = dn_meta["gt_boxes"]                # [B, dn_len, 4]

        if valid.sum() == 0:
            z = torch.tensor(0.0, device=dn_pred_logits.device)
            return z, z, z

        num_valid = int(valid.sum())

        # Classification
        pred_cls = dn_pred_logits[valid]   # [K, C+1]
        tgt_cls = gt_labels[valid]         # [K]
        loss_dn_ce = F.cross_entropy(pred_cls, tgt_cls, weight=self.empty_weight)

        # Box regression
        pred_box = dn_pred_boxes[valid]    # [K, 4]
        tgt_box = gt_boxes[valid]          # [K, 4]
        loss_dn_bbox = F.l1_loss(pred_box, tgt_box, reduction="sum") / max(num_valid, 1)

        pred_xyxy = box_cxcywh_to_xyxy(pred_box)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_box)
        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        loss_dn_giou = (1 - giou.diag()).sum() / max(num_valid, 1)

        return loss_dn_ce, loss_dn_bbox, loss_dn_giou

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from DETRHead + optional MaskBranch
            targets: list[dict] with 'labels', 'boxes', optional 'masks'
        Returns:
            dict of losses including 'loss_total'
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, targets)
        num_objects = sum(len(t["labels"]) for t in targets)

        # Matching losses
        loss_ce = self.loss_labels(pred_logits, targets, indices, num_objects)
        loss_bbox, loss_giou = self.loss_boxes(pred_boxes, targets, indices, num_objects)

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

        # Mask losses (segment task only)
        if "pred_masks" in outputs:
            loss_mask, loss_dice = self.loss_masks(
                outputs["pred_masks"], targets, indices, num_objects,
            )
            losses["loss_mask"] = loss_mask
            losses["loss_dice"] = loss_dice

        # DN losses (training with denoising)
        if "dn_pred_logits" in outputs and outputs.get("dn_meta") is not None:
            loss_dn_ce, loss_dn_bbox, loss_dn_giou = self.loss_dn(
                outputs["dn_pred_logits"], outputs["dn_pred_boxes"],
                outputs["dn_meta"],
            )
            losses["loss_dn_ce"] = loss_dn_ce
            losses["loss_dn_bbox"] = loss_dn_bbox
            losses["loss_dn_giou"] = loss_dn_giou

        # Weighted total
        loss_total = torch.tensor(0.0, device=pred_logits.device)
        for k, v in losses.items():
            w = self.weight_dict.get(k, 0.0)
            if w > 0:
                loss_total = loss_total + w * v
        losses["loss_total"] = loss_total

        if DEBUG:
            print("[DETR Loss]", {k: f"{v.item():.4f}" for k, v in losses.items()})

        return losses

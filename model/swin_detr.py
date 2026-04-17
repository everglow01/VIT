"""
Swin Transformer + DN-DETR detection / instance segmentation model.

Unified entry point that wires together:
  - Swin backbone  (frozen feature extractor)
  - DETRHead       (Transformer encoder-decoder with DN denoising)
  - MaskBranch     (optional, for segment task)
  - SetCriterion   (training loss)

Interface mirrors ViTFasterRCNN / ViTMaskRCNN so that existing
train.py / predict.py evaluation helpers can be reused.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import model.swin_model as swin_models
from model.detr_head import DETRHead
from model.mask_branch import MaskBranch
from tools.matcher import HungarianMatcher, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from tools.detr_loss import SetCriterion


class SwinDETR(nn.Module):
    """
    Full Swin + DN-DETR model for detection or instance segmentation.

    Training:
        model(images, targets) -> dict of losses  (includes 'loss_total')
    Inference:
        model(images) -> list[dict] with 'boxes', 'scores', 'labels', ['masks']
    """
    def __init__(
        self,
        backbone_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_queries=100,
        dropout=0.1,
        num_dn_groups=2,
        task="detect",
        pretrained_backbone=None,
        freeze_backbone=True,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        eos_coef=0.1,
        min_size=800,
        max_size=1333,
        conf_thresh=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.task = task
        self.conf_thresh = conf_thresh

        # ---- Backbone ----
        factory = getattr(swin_models, backbone_name)
        self.backbone = factory(num_classes=0)
        embed_dim = self.backbone.embed_dim

        if pretrained_backbone:
            ckpt = torch.load(pretrained_backbone, map_location="cpu")
            state = ckpt.get("model", ckpt.get("model_state", ckpt))
            state = {k: v for k, v in state.items()
                     if not k.startswith("head.") and not k.startswith("norm_cls.")}
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"[Swin] missing keys ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                print(f"[Swin] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

        # freeze_backbone: True/"all" → freeze all stages
        #                  "partial"   → freeze stage0+1 (patch embed, layers.0, layers.1)
        #                               unfreeze stage2+3 (layers.2, layers.3) + all norms
        #                  False/"none"→ train full backbone
        _freeze = freeze_backbone
        if _freeze is True or _freeze == "all":
            for name, p in self.backbone.named_parameters():
                if name.startswith(("norm0", "norm1", "norm2", "norm3")):
                    continue  # per-stage norms are randomly initialised → keep trainable
                p.requires_grad_(False)
        elif _freeze == "partial":
            for name, p in self.backbone.named_parameters():
                # Keep stage2 (layers.2), stage3 (layers.3) and all norms trainable
                if any(name.startswith(x) for x in
                       ("layers.2", "layers.3", "norm0", "norm1", "norm2", "norm3")):
                    continue
                p.requires_grad_(False)
            print("[SwinDETR] partial freeze: stage0+1 frozen, stage2+3 trainable")

        # Multi-scale channel sizes: p3=2C, p4=4C, p5=8C
        in_channels_list = [embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # ---- DETR Head ----
        self.detr_head = DETRHead(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            dropout=dropout,
            num_dn_groups=num_dn_groups,
        )

        # ---- Mask Branch (segment task only) ----
        self.mask_branch = None
        if task == "segment":
            self.mask_branch = MaskBranch(d_model=d_model, mask_dim=d_model)

        # ---- Loss ----
        matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
        )
        weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": cost_bbox,
            "loss_giou": cost_giou,
            "loss_dn_ce": 0.5,
            "loss_dn_bbox": cost_bbox * 0.5,
            "loss_dn_giou": cost_giou * 0.5,
        }
        if task == "segment":
            weight_dict["loss_mask"] = 2.0
            weight_dict["loss_dice"] = 2.0

        # Auxiliary decoder losses: one set per intermediate decoder layer.
        # Use the same weights as the main output so every layer receives equal supervision.
        for i in range(num_decoder_layers - 1):
            weight_dict[f"loss_ce_aux_{i}"]   = 1.0
            weight_dict[f"loss_bbox_aux_{i}"] = cost_bbox
            weight_dict[f"loss_giou_aux_{i}"] = cost_giou

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
        )

        # ---- Image transform ----
        self.transform = GeneralizedRCNNTransform(
            min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SwinDETR] params: {total:,} total, {trainable:,} trainable")

    # ------------------------------------------------------------------
    # Target format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_targets_for_detr(targets, image_sizes):
        """Convert torchvision-style targets to DETR format.

        torchvision: labels 1-based, boxes x1y1x2y2 absolute
        DETR:        labels 0-based, boxes cxcywh normalised [0,1]
        """
        detr_targets = []
        for tgt, (h, w) in zip(targets, image_sizes):
            boxes = tgt["boxes"].clone()                  # [M, 4] x1y1x2y2
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
            boxes_cxcywh[:, 0::2] /= w
            boxes_cxcywh[:, 1::2] /= h

            dt = {
                "labels": tgt["labels"] - 1,              # 1-based → 0-based
                "boxes": boxes_cxcywh,
            }
            if "masks" in tgt:
                dt["masks"] = tgt["masks"].float()
            if "image_id" in tgt:
                dt["image_id"] = tgt["image_id"]

            detr_targets.append(dt)
        return detr_targets

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images, targets=None):
        """
        Args:
            images:  list[Tensor [C,H,W]]
            targets: list[dict]  (training only)
        Returns:
            training → dict of losses (includes 'loss_total')
            eval     → list[dict] with 'boxes', 'scores', 'labels', ['masks']
        """
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        images_t, targets_t = self.transform(images, targets)

        # Backbone → multi-scale features
        features = self.backbone.forward_features_map(images_t.tensors)

        # ==================== Training ====================
        if self.training:
            detr_targets = self._prepare_targets_for_detr(
                targets_t, images_t.image_sizes,
            )

            outputs = self.detr_head(features, detr_targets)

            if self.mask_branch is not None:
                pred_masks = self.mask_branch(
                    outputs["hs"], outputs["memory"], outputs["spatial_shapes"],
                )
                outputs["pred_masks"] = pred_masks

            return self.criterion(outputs, detr_targets)

        # ==================== Inference ====================
        outputs = self.detr_head(features, targets=None)

        pred_logits = outputs["pred_logits"]       # [B, nq, C+1]
        pred_boxes = outputs["pred_boxes"]         # [B, nq, 4]
        B = pred_logits.shape[0]

        results = []
        keeps = []                                 # store per-image keep mask for masks

        for b in range(B):
            probs = pred_logits[b].softmax(-1)     # [nq, C+1]
            scores, labels = probs[:, :-1].max(-1) # exclude background column
            keep = scores > self.conf_thresh
            keeps.append(keep)

            s = scores[keep]
            l = labels[keep] + 1                   # 0-based → 1-based

            boxes = pred_boxes[b, keep]
            h, w = images_t.image_sizes[b]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            boxes_xyxy[:, 0::2] *= w
            boxes_xyxy[:, 1::2] *= h

            results.append({"boxes": boxes_xyxy, "scores": s, "labels": l})

        # Postprocess boxes (rescale to original image size)
        results = self.transform.postprocess(
            results, images_t.image_sizes, original_image_sizes,
        )

        # Mask prediction & resize to original image
        if self.mask_branch is not None:
            device = pred_logits.device
            for b in range(B):
                keep = keeps[b]
                orig_h, orig_w = original_image_sizes[b]
                if keep.sum() > 0:
                    pred_masks = self.mask_branch(
                        outputs["hs"][b:b + 1, keep],
                        outputs["memory"][b:b + 1],
                        outputs["spatial_shapes"],
                    )  # [1, K, H3, W3]
                    pred_masks = F.interpolate(
                        pred_masks[0].unsqueeze(1).float(),   # [K, 1, H3, W3]
                        size=(orig_h, orig_w),
                        mode="bilinear", align_corners=False,
                    ).sigmoid()                                # [K, 1, H_orig, W_orig]
                    results[b]["masks"] = pred_masks
                else:
                    results[b]["masks"] = torch.zeros(
                        (0, 1, orig_h, orig_w), device=device,
                    )

        return results


# ======================================================================
# Factory
# ======================================================================

def build_detr_model(
    backbone_name,
    num_classes,
    task="detect",
    backbone_weights="",
    freeze_backbone=True,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_queries=100,
    num_dn_groups=2,
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0,
    min_size=800,
    max_size=1333,
    conf_thresh=0.5,
):
    return SwinDETR(
        backbone_name=backbone_name,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_queries=num_queries,
        num_dn_groups=num_dn_groups,
        task=task,
        pretrained_backbone=backbone_weights if backbone_weights else None,
        freeze_backbone=freeze_backbone,
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        min_size=min_size,
        max_size=max_size,
        conf_thresh=conf_thresh,
    )

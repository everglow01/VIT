"""
ViT + ViTDet-style FPN + Faster R-CNN detection head.

ViTDet FPN: single-scale ViT output [B, D, 14, 14] -> multi-scale {P2,P3,P4,P5}
via deconv/conv, then fed into torchvision FPN lateral fusion.
Faster R-CNN head: torchvision RPN + RoIAlign + box/cls branches.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import model.vit_model as vit_models


class ViTDetFPN(nn.Module):
    """
    Converts single-scale ViT patch feature map [B, D, 14, 14] into
    4-level FPN output: P2(56x56), P3(28x28), P4(14x14), P5(7x7).
    """
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        # P2: 14->56 via 2x deconv chain
        self.to_p2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels // 2, out_channels, 2, stride=2),
        )
        # P3: 14->28 via 1x deconv
        self.to_p3 = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # P4: 14->14 via 1x1 conv
        self.to_p4 = nn.Conv2d(in_channels, out_channels, 1)
        # P5: 14->7 via stride-2 conv
        self.to_p5 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

        # torchvision FPN for lateral + top-down fusion
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels, out_channels, out_channels, out_channels],
            out_channels=out_channels,
        )

    def forward(self, x):
        # x: [B, D, 14, 14]
        p2 = self.to_p2(x)   # [B, 256, 56, 56]
        p3 = self.to_p3(x)   # [B, 256, 28, 28]
        p4 = self.to_p4(x)   # [B, 256, 14, 14]
        p5 = self.to_p5(x)   # [B, 256, 7,  7]

        feat_dict = OrderedDict([("p2", p2), ("p3", p3), ("p4", p4), ("p5", p5)])
        return self.fpn(feat_dict)  # same keys, fused


class FasterRCNNHead(nn.Module):
    """Two-layer MLP box/cls head used after RoI pooling."""
    def __init__(self, in_channels: int, representation_size: int, num_classes: int):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        return self.cls_score(x), self.bbox_pred(x)


class ViTFasterRCNN(nn.Module):
    """
    Full detection model: ViT backbone + ViTDet FPN + Faster R-CNN.

    Args:
        backbone_name: factory function name in vit_model.py
        num_classes: number of foreground classes (background added internally)
        backbone_weights: path to pretrained ViT weights, or ''
        freeze_backbone: if True, freeze ViT parameters
        fpn_out_channels: FPN output channels (default 256)
        roi_output_size: RoI pooling output size (default 7)
        representation_size: FC head hidden size (default 1024)
    """
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224_in21k",
        num_classes: int = 91,
        backbone_weights: str = "",
        freeze_backbone: bool = True,
        fpn_out_channels: int = 256,
        roi_output_size: int = 7,
        representation_size: int = 1024,
    ):
        super().__init__()

        # --- backbone ---
        factory = getattr(vit_models, backbone_name)
        self.backbone = factory(num_classes=0)  # num_classes=0 -> head is Identity
        embed_dim = self.backbone.embed_dim

        if backbone_weights:
            import torch
            ckpt = torch.load(backbone_weights, map_location="cpu")
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            # strip head keys
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            self.backbone.load_state_dict(state, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # --- neck ---
        self.neck = ViTDetFPN(in_channels=embed_dim, out_channels=fpn_out_channels)

        # --- RPN ---
        anchor_sizes = ((32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(fpn_out_channels, anchor_gen.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_gen,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )

        # --- RoI head ---
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=roi_output_size,
            sampling_ratio=2,
        )
        in_channels_roi = fpn_out_channels * roi_output_size * roi_output_size
        box_head = FasterRCNNHead(in_channels_roi, representation_size, num_classes + 1)

        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=None,          # predictor is inside box_head
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )

        # --- image normalization transform (ViT uses 0.5/0.5) ---
        self.transform = GeneralizedRCNNTransform(
            min_size=224, max_size=224,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            fixed_size=(224, 224),
        )

    def forward(self, images, targets=None):
        """
        images: list of Tensor [C, H, W]
        targets (training): list of dict {boxes: [N,4], labels: [N]}
        Returns:
          training -> dict of losses
          eval     -> list of dict {boxes, labels, scores}
        """
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        images, targets = self.transform(images, targets)

        # backbone: batch tensor -> feature map
        img_tensor = images.tensors                          # [B, 3, H, W]
        feat_map = self.backbone.forward_features_map(img_tensor)  # [B, D, 14, 14]

        # neck: single-scale -> multi-scale FPN dict
        features = self.neck(feat_map)                       # {p2,p3,p4,p5}

        # RPN
        proposals, rpn_losses = self.rpn(images, features, targets)

        # RoI head — needs box_head to return (cls_logits, bbox_pred) tuple
        detections, detector_losses = self._roi_forward(features, proposals, images.image_sizes, targets)

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

    def _roi_forward(self, features, proposals, image_shapes, targets):
        """
        Wraps RoIHeads.forward() but patches box_predictor to use our combined head.
        We monkey-patch by temporarily setting box_predictor to a wrapper.
        """
        # torchvision RoIHeads expects box_head to output flat features,
        # and box_predictor to output (cls, bbox). We combine them in FasterRCNNHead.
        # Patch: set box_predictor to identity, override box_head to return both.
        class _SplitPredictor(nn.Module):
            def __init__(self, combined_head):
                super().__init__()
                self.combined_head = combined_head
            def forward(self, x):
                return self.combined_head(x)

        # Temporarily replace
        orig_predictor = self.roi_heads.box_predictor
        self.roi_heads.box_predictor = _SplitPredictor(self.roi_heads.box_head)
        # box_head becomes identity (predictor does the work)
        orig_box_head = self.roi_heads.box_head
        self.roi_heads.box_head = nn.Identity()

        result, losses = self.roi_heads(features, proposals, image_shapes, targets)

        # Restore
        self.roi_heads.box_head = orig_box_head
        self.roi_heads.box_predictor = orig_predictor
        return result, losses


def build_detection_model(
    backbone_name: str,
    num_classes: int,
    backbone_weights: str = "",
    freeze_backbone: bool = True,
) -> ViTFasterRCNN:
    return ViTFasterRCNN(
        backbone_name=backbone_name,
        num_classes=num_classes,
        backbone_weights=backbone_weights,
        freeze_backbone=freeze_backbone,
    )

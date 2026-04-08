"""
ViT/Swin + ViTDet/SwinFPN + Mask R-CNN segmentation head.
Reuses ViTDetFPN, SwinFPN, FasterRCNNHead, _is_swin from detection_head.py.
"""
import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor

from model.detection_head import ViTDetFPN, SwinFPN, FasterRCNNHead, _is_swin
import model.vit_model as vit_models
import model.swin_model as swin_models


class ViTMaskRCNN(nn.Module):
    """
    Full instance segmentation model: ViT backbone + ViTDet FPN + Mask R-CNN.

    Args:
        backbone_name: factory function name in vit_model.py
        num_classes: number of foreground classes (background added internally)
        backbone_weights: path to pretrained ViT weights, or ''
        freeze_backbone: if True, freeze ViT parameters
        fpn_out_channels: FPN output channels (default 256)
        roi_output_size: RoI pooling output size for box head (default 7)
        mask_roi_output_size: RoI pooling output size for mask head (default 14)
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
        mask_roi_output_size: int = 14,
        representation_size: int = 1024,
    ):
        super().__init__()

        # --- backbone ---
        if _is_swin(backbone_name):
            factory = getattr(swin_models, backbone_name)
            self.backbone = factory(num_classes=0)
            if backbone_weights:
                ckpt = torch.load(backbone_weights, map_location="cpu")
                state = ckpt.get("model", ckpt.get("model_state", ckpt))
                state = {k: v for k, v in state.items()
                         if not k.startswith("head.") and not k.startswith("norm_cls.")}
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                if missing:
                    print(f"[Swin] missing keys ({len(missing)}): {missing[:3]} ...")
                if unexpected:
                    print(f"[Swin] unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
            image_mean, image_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            factory = getattr(vit_models, backbone_name)
            self.backbone = factory(num_classes=0)
            if backbone_weights:
                ckpt = torch.load(backbone_weights, map_location="cpu")
                state = ckpt["model_state"] if "model_state" in ckpt else ckpt
                state = {k: v for k, v in state.items() if not k.startswith("head.")}
                self.backbone.load_state_dict(state, strict=False)
            image_mean, image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if _is_swin(backbone_name) and name.startswith(("norm0", "norm1", "norm2", "norm3")):
                    continue
                p.requires_grad_(False)

        # --- neck ---
        if _is_swin(backbone_name):
            self.neck = SwinFPN(embed_dim=embed_dim, out_channels=fpn_out_channels)
        else:
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

        # --- box RoI head ---
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=roi_output_size,
            sampling_ratio=2,
        )
        in_channels_roi = fpn_out_channels * roi_output_size * roi_output_size
        box_head = FasterRCNNHead(in_channels_roi, representation_size, num_classes + 1)

        # --- mask RoI head ---
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=mask_roi_output_size,
            sampling_ratio=2,
        )
        # 4-layer conv head + deconv predictor (standard Mask R-CNN)
        mask_head = MaskRCNNHeads(fpn_out_channels, (256, 256, 256, 256), 1)
        mask_predictor = MaskRCNNPredictor(256, 256, num_classes + 1)

        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=None,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
        )

        self.transform = GeneralizedRCNNTransform(
            min_size=224, max_size=224,
            image_mean=image_mean,
            image_std=image_std,
            fixed_size=(224, 224),
        )

    def forward(self, images, targets=None):
        """
        images: list of Tensor [C, H, W]
        targets (training): list of dict {boxes: [N,4], labels: [N], masks: [N,H,W]}
        Returns:
          training -> dict of losses
          eval     -> list of dict {boxes, labels, scores, masks}
        """
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        images, targets = self.transform(images, targets)

        # backbone -> neck -> FPN dict {p2,p3,p4,p5}
        features = self.neck(self.backbone.forward_features_map(images.tensors))

        proposals, rpn_losses = self.rpn(images, features, targets)

        # Patch box_predictor same as detection_head
        class _SplitPredictor(nn.Module):
            def __init__(self, combined_head):
                super().__init__()
                self.combined_head = combined_head
            def forward(self, x):
                return self.combined_head(x)

        orig_predictor = self.roi_heads.box_predictor
        self.roi_heads.box_predictor = _SplitPredictor(self.roi_heads.box_head)
        orig_box_head = self.roi_heads.box_head
        self.roi_heads.box_head = nn.Identity()

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        self.roi_heads.box_head = orig_box_head
        self.roi_heads.box_predictor = orig_predictor

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections


def build_segmentation_model(
    backbone_name: str,
    num_classes: int,
    backbone_weights: str = "",
    freeze_backbone: bool = True,
) -> ViTMaskRCNN:
    return ViTMaskRCNN(
        backbone_name=backbone_name,
        num_classes=num_classes,
        backbone_weights=backbone_weights,
        freeze_backbone=freeze_backbone,
    )

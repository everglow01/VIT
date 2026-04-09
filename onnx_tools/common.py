"""Shared utilities for ONNX export and verification scripts."""
import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import model.vit_model as vit_models
import model.swin_model as swin_models
from model.detection_head import build_detection_model
from model.segmentation_head import build_segmentation_model

_MODEL_NAMES = [
    "vit_base_patch16_224_in21k",
    "vit_base_patch32_224_in21k",
    "vit_large_patch16_224_in21k",
    "vit_large_patch32_224_in21k",
    "vit_huge_patch14_224_in21k",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_base_patch4_window12_384",
    "swin_large_patch4_window7_224",
    "swin_large_patch4_window12_384",
]

MODEL_BUILDERS = {
    n: fn
    for n in _MODEL_NAMES
    if callable(fn := getattr(vit_models, n, None) or getattr(swin_models, n, None))
}


def detect_task(state_dict: dict) -> str:
    """Infer task type from checkpoint keys: 'classify', 'detect', or 'segment'."""
    if not isinstance(state_dict, dict):
        return "classify"
    has_det = has_seg = False
    for k in state_dict:
        if isinstance(k, str):
            if k.startswith(("backbone.", "neck.", "rpn.", "roi_heads.")):
                has_det = True
            if "mask_head" in k:
                has_seg = True
    if has_seg:
        return "segment"
    if has_det:
        return "detect"
    return "classify"


def load_state(weights_path: str, device: torch.device) -> dict:
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in state:
                return state[key]
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format: expected state_dict or checkpoint dict.")
    return state


def build_model(task: str, backbone: str, num_classes: int, device: torch.device) -> nn.Module:
    if task == "classify":
        return MODEL_BUILDERS[backbone](num_classes=num_classes).to(device)
    if task == "detect":
        return build_detection_model(backbone, num_classes, backbone_weights="", freeze_backbone=False).to(device)
    return build_segmentation_model(backbone, num_classes, backbone_weights="", freeze_backbone=False).to(device)


class DetectionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        out = self.model([x[0]])[0]
        return out["boxes"], out["scores"], out["labels"]


class SegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        out = self.model([x[0]])[0]
        masks = out.get("masks", torch.zeros(0, 1, 1, 1, device=x.device))
        return out["boxes"], out["scores"], out["labels"], masks


TASK_OUTPUT_NAMES = {
    "classify": (["images"], ["logits"],
                 {"images": {0: "batch"}, "logits": {0: "batch"}}),
    "detect":   (["images"], ["boxes", "scores", "labels"],
                 {"images": {0: "batch", 2: "height", 3: "width"},
                  "boxes": {0: "num_det"}, "scores": {0: "num_det"}, "labels": {0: "num_det"}}),
    "segment":  (["images"], ["boxes", "scores", "labels", "masks"],
                 {"images": {0: "batch", 2: "height", 3: "width"},
                  "boxes": {0: "num_det"}, "scores": {0: "num_det"},
                  "labels": {0: "num_det"}, "masks": {0: "num_det"}}),
}

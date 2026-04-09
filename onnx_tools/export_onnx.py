import argparse
import os
import sys
import torch
import torch.nn as nn

# Allow running as: python3 onnx_tools/export_onnx.py ...
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import model.vit_model as vit_models
import model.swin_model as swin_models
from model.detection_head import build_detection_model
from model.segmentation_head import build_segmentation_model


def _build_model_registry():
    names = [
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
    registry = {}
    for n in names:
        fn = getattr(vit_models, n, None) or getattr(swin_models, n, None)
        if callable(fn):
            registry[n] = fn
    return registry


MODEL_BUILDERS = _build_model_registry()


# ── ONNX wrappers for detection / segmentation ──────────────────────────────

class DetectionWrapper(nn.Module):
    """Unwrap torchvision detection output for ONNX (batch=1)."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        out = self.model([x[0]])[0]
        return out["boxes"], out["scores"], out["labels"]


class SegmentationWrapper(nn.Module):
    """Unwrap torchvision Mask R-CNN output for ONNX (batch=1)."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        out = self.model([x[0]])[0]
        masks = out.get("masks", torch.zeros(0, 1, 1, 1, device=x.device))
        return out["boxes"], out["scores"], out["labels"], masks


# ── checkpoint helpers ───────────────────────────────────────────────────────

def _unwrap_state(state):
    for key in ("model", "model_state", "state_dict"):
        if isinstance(state, dict) and key in state:
            return state[key]
    return state


def _is_detseg_checkpoint(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(
        isinstance(k, str) and k.startswith(("backbone.", "neck.", "rpn.", "roi_heads."))
        for k in state_dict
    )


def _is_seg_checkpoint(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(isinstance(k, str) and "mask_head" in k for k in state_dict)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export ViT/Swin classification/detection/segmentation model to ONNX")
    parser.add_argument("--weights", "--weight", dest="weights", type=str, required=True,
                        help="Path to .pt or .pth model file")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224_in21k",
                        choices=list(MODEL_BUILDERS.keys()), help="Model architecture name")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "classify", "detect", "segment"],
                        help="Task type (default: auto-detect from checkpoint)")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of foreground classes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: same folder as weights)")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )

    raw = torch.load(args.weights, map_location=device)
    state = _unwrap_state(raw)

    # auto-detect task
    task = args.task
    if task == "auto":
        if _is_detseg_checkpoint(state):
            task = "segment" if _is_seg_checkpoint(state) else "detect"
        else:
            task = "classify"
    print(f"Task: {task}")

    # build model
    if task == "classify":
        net = MODEL_BUILDERS[args.model](num_classes=args.num_classes).to(device)
        net.load_state_dict(state, strict=False)
        net.eval()
        export_model = net
        input_names, output_names = ["images"], ["logits"]
        dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}

    elif task == "detect":
        net = build_detection_model(
            backbone_name=args.model, num_classes=args.num_classes,
            backbone_weights="", freeze_backbone=False,
        ).to(device)
        net.load_state_dict(state, strict=False)
        export_model = DetectionWrapper(net).to(device)
        input_names = ["images"]
        output_names = ["boxes", "scores", "labels"]
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "num_det"}, "scores": {0: "num_det"}, "labels": {0: "num_det"},
        }

    else:  # segment
        net = build_segmentation_model(
            backbone_name=args.model, num_classes=args.num_classes,
            backbone_weights="", freeze_backbone=False,
        ).to(device)
        net.load_state_dict(state, strict=False)
        export_model = SegmentationWrapper(net).to(device)
        input_names = ["images"]
        output_names = ["boxes", "scores", "labels", "masks"]
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "num_det"}, "scores": {0: "num_det"},
            "labels": {0: "num_det"}, "masks": {0: "num_det"},
        }

    out_path = args.output or (os.path.splitext(args.weights)[0] + ".onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    torch.onnx.export(
        export_model, dummy, out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()

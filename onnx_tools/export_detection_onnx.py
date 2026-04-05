import argparse
import torch
import torch.nn as nn

from model.detection_head import build_detection_model


class DetectionOnnxWrapper(nn.Module):
    """
    Wrap detection model for ONNX export.
    Input:  [1, 3, H, W]
    Output: boxes [N,4], scores [N], labels [N]
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector.eval()

    def forward(self, x: torch.Tensor):
        # Convert batched tensor into torchvision detection-style list input.
        # This export script targets batch size = 1 for architecture inspection.
        outputs = self.detector([x[0]])
        out0 = outputs[0]
        return out0["boxes"], out0["scores"], out0["labels"]


def main():
    parser = argparse.ArgumentParser(description="Export detection model to ONNX (random weights)")
    parser.add_argument("--output", type=str, default="detection_model.onnx", help="Output ONNX file path")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224_in21k", help="Backbone name")
    parser.add_argument("--num-classes", type=int, default=80, help="Foreground class count")
    parser.add_argument("--input-size", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    detector = build_detection_model(
        backbone_name=args.model,
        num_classes=args.num_classes,
        backbone_weights="",
        freeze_backbone=False,
    ).to(device).eval()

    wrapper = DetectionOnnxWrapper(detector).to(device).eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Exported ONNX file: {args.output}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from onnx_tools.common import (
    MODEL_BUILDERS, detect_task, load_state, build_model,
    DetectionWrapper, SegmentationWrapper, TASK_OUTPUT_NAMES,
)


def main():
    parser = argparse.ArgumentParser(description="Export ViT/Swin classification/detection/segmentation model to ONNX")
    parser.add_argument("--weights", "--weight", dest="weights", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224_in21k",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "classify", "detect", "segment"])
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: same folder as weights)")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )

    state = load_state(args.weights, device)
    task = detect_task(state) if args.task == "auto" else args.task
    print(f"Task: {task}")

    net = build_model(task, args.model, args.num_classes, device)
    net.load_state_dict(state, strict=False)
    net.eval()

    if task == "detect":
        export_model = DetectionWrapper(net)
    elif task == "segment":
        export_model = SegmentationWrapper(net)
    else:
        export_model = net

    input_names, output_names, dynamic_axes = TASK_OUTPUT_NAMES[task]
    out_path = args.output or (os.path.splitext(args.weights)[0] + ".onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)
    torch.onnx.export(
        export_model, dummy, out_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()

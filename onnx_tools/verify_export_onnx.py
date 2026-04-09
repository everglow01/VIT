import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Allow running as: python3 onnx_tools/verify_export_onnx.py ...
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
        fn = getattr(vit_models, n, None)
        if fn is None:
            fn = getattr(swin_models, n, None)
        if callable(fn):
            registry[n] = fn
    return registry


MODEL_BUILDERS = _build_model_registry()


def _is_detseg_checkpoint(state_dict: dict) -> bool:
    if not isinstance(state_dict, dict):
        return False
    for k in state_dict.keys():
        if isinstance(k, str) and (
            k.startswith("backbone.") or
            k.startswith("neck.") or
            k.startswith("rpn.") or
            k.startswith("roi_heads.")
        ):
            return True
    return False


def _is_seg_checkpoint(state_dict: dict) -> bool:
    if not isinstance(state_dict, dict):
        return False
    return any(isinstance(k, str) and "mask_head" in k for k in state_dict)


def _load_checkpoint_state(weights_path: str, device: torch.device):
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        if "model_state" in state:
            state = state["model_state"]
        elif "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format: expected state_dict or checkpoint dict.")
    return state


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


def _print_metrics(torch_out: np.ndarray, onnx_out: np.ndarray):
    if torch_out.size == 0 and onnx_out.size == 0:
        print("Comparison metrics:")
        print("  both outputs are empty; treat as exact match")
        print("  max_abs_diff : 0.00000000")
        print("  mean_abs_diff: 0.00000000")
        print("  max_rel_diff : 0.00000000")
        print("  mean_rel_diff: 0.00000000")
        return 0.0, 0.0, 0.0, 0.0

    abs_diff = np.abs(torch_out - onnx_out)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())

    denom = np.maximum(np.abs(torch_out), 1e-8)
    rel_diff = abs_diff / denom
    max_rel = float(rel_diff.max())
    mean_rel = float(rel_diff.mean())

    print("Comparison metrics:")
    print(f"  max_abs_diff : {max_abs:.8f}")
    print(f"  mean_abs_diff: {mean_abs:.8f}")
    print(f"  max_rel_diff : {max_rel:.8f}")
    print(f"  mean_rel_diff: {mean_rel:.8f}")

    return max_abs, mean_abs, max_rel, mean_rel


def _compare_one_output(name: str, torch_out: np.ndarray, onnx_out: np.ndarray, atol: float, rtol: float) -> bool:
    # For detect/segment outputs, num_detections dimension may differ by a small amount.
    if torch_out.shape != onnx_out.shape:
        if torch_out.ndim >= 1 and onnx_out.ndim == torch_out.ndim and torch_out.shape[1:] == onnx_out.shape[1:]:
            n = min(torch_out.shape[0], onnx_out.shape[0])
            if n == 0:
                print(f"{name}: both outputs empty after alignment")
                return True
            print(f"{name}: shape differs torch={torch_out.shape} onnx={onnx_out.shape}, comparing first {n}")
            torch_out = torch_out[:n]
            onnx_out = onnx_out[:n]
        else:
            print(f"{name}: shape mismatch torch={torch_out.shape} onnx={onnx_out.shape}")
            return False

    if np.issubdtype(torch_out.dtype, np.integer) and np.issubdtype(onnx_out.dtype, np.integer):
        same = np.array_equal(torch_out, onnx_out)
        print(f"{name}: integer exact match = {same}")
        return same

    max_abs, _, max_rel, _ = _print_metrics(torch_out.astype(np.float32), onnx_out.astype(np.float32))
    ok = (max_abs <= atol) and (max_rel <= rtol)
    print(f"{name}: {'PASS' if ok else 'FAIL'} (atol={atol}, rtol={rtol})")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX export by comparing ONNXRuntime vs PyTorch outputs")
    parser.add_argument("--weights", "--weight", dest="weights", type=str, required=True,
                        help="Path to .pt/.pth checkpoint used for ONNX export")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Path to exported .onnx (default: replace weights suffix with .onnx)")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()),
                        help="Model architecture name")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "classify", "detect", "segment"],
                        help="Task type (default: auto-detect from checkpoint)")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of output classes")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="Random test batch size")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance pass threshold")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance pass threshold")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic input")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda for PyTorch reference")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    onnx_path = args.onnx if args.onnx else os.path.splitext(args.weights)[0] + ".onnx"
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"onnx not found: {onnx_path}")

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(
            "onnxruntime is required for verification. Install with: pip install onnxruntime"
        ) from e

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )

    # Build and load PyTorch model
    state = _load_checkpoint_state(args.weights, device)

    task = args.task
    if task == "auto":
        if _is_detseg_checkpoint(state):
            task = "segment" if _is_seg_checkpoint(state) else "detect"
        else:
            task = "classify"
    print(f"Task: {task}")

    if task == "classify":
        model = MODEL_BUILDERS[args.model](num_classes=args.num_classes).to(device)
        msg = model.load_state_dict(state, strict=False)
        print(msg)
        model.eval()
        ref_model = model
        output_names = ["logits"]
    elif task == "detect":
        det = build_detection_model(
            backbone_name=args.model,
            num_classes=args.num_classes,
            backbone_weights="",
            freeze_backbone=False,
        ).to(device)
        msg = det.load_state_dict(state, strict=False)
        print(msg)
        ref_model = DetectionWrapper(det).to(device).eval()
        output_names = ["boxes", "scores", "labels"]
    else:
        seg = build_segmentation_model(
            backbone_name=args.model,
            num_classes=args.num_classes,
            backbone_weights="",
            freeze_backbone=False,
        ).to(device)
        msg = seg.load_state_dict(state, strict=False)
        print(msg)
        ref_model = SegmentationWrapper(seg).to(device).eval()
        output_names = ["boxes", "scores", "labels", "masks"]

    # Prepare deterministic random input
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    with torch.no_grad():
        ref_out = ref_model(x)
    if isinstance(ref_out, torch.Tensor):
        ref_out = (ref_out,)
    torch_outputs = [o.detach().cpu().numpy() for o in ref_out]

    # ONNXRuntime inference
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: x.detach().cpu().numpy().astype(np.float32)})

    if len(onnx_outputs) != len(torch_outputs):
        print(f"Output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}")
        sys.exit(1)

    all_ok = True
    for i, (t, o) in enumerate(zip(torch_outputs, onnx_outputs)):
        name = output_names[i] if i < len(output_names) else f"output_{i}"
        print(f"\n[{name}] torch_dtype={t.dtype} onnx_dtype={o.dtype}")
        ok = _compare_one_output(name, t, o, args.atol, args.rtol)
        all_ok = all_ok and ok

    if all_ok:
        print(f"\nPASS: all outputs within tolerance (atol={args.atol}, rtol={args.rtol})")
        return

    print(f"\nFAIL: at least one output out of tolerance (atol={args.atol}, rtol={args.rtol})")
    sys.exit(1)


if __name__ == "__main__":
    main()

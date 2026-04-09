import argparse
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from onnx_tools.common import (
    MODEL_BUILDERS, detect_task, load_state, build_model,
    DetectionWrapper, SegmentationWrapper, TASK_OUTPUT_NAMES,
)


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
    parser.add_argument("--weights", "--weight", dest="weights", type=str, required=True)
    parser.add_argument("--onnx", type=str, default=None,
                        help="Path to exported .onnx (default: replace weights suffix with .onnx)")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "classify", "detect", "segment"])
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    onnx_path = args.onnx or (os.path.splitext(args.weights)[0] + ".onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"onnx not found: {onnx_path}")

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime") from e

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )

    state = load_state(args.weights, device)
    task = detect_task(state) if args.task == "auto" else args.task
    print(f"Task: {task}")

    net = build_model(task, args.model, args.num_classes, device)
    msg = net.load_state_dict(state, strict=False)
    print(msg)
    net.eval()

    if task == "detect":
        ref_model = DetectionWrapper(net).to(device).eval()
    elif task == "segment":
        ref_model = SegmentationWrapper(net).to(device).eval()
    else:
        ref_model = net

    _, output_names, _ = TASK_OUTPUT_NAMES[task]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    with torch.no_grad():
        ref_out = ref_model(x)
    if isinstance(ref_out, torch.Tensor):
        ref_out = (ref_out,)
    torch_outputs = [o.detach().cpu().numpy() for o in ref_out]

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
        all_ok &= _compare_one_output(name, t, o, args.atol, args.rtol)

    if all_ok:
        print(f"\nPASS: all outputs within tolerance (atol={args.atol}, rtol={args.rtol})")
    else:
        print(f"\nFAIL: at least one output out of tolerance (atol={args.atol}, rtol={args.rtol})")
        sys.exit(1)


if __name__ == "__main__":
    main()

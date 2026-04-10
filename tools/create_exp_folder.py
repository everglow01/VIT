import os
import re


def _next_exp_dir(parent: str) -> str:
    os.makedirs(parent, exist_ok=True)
    existing = os.listdir(parent)
    # Match "exp" (index 0) and "expN" (index N)
    max_idx = -1
    for name in existing:
        if name == "exp":
            max_idx = max(max_idx, 0)
        else:
            m = re.fullmatch(r"exp(\d+)", name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

    if max_idx < 0:
        path = os.path.join(parent, "exp")
    else:
        path = os.path.join(parent, f"exp{max_idx + 1}")
    os.makedirs(path)
    return path


def create_exp_folder():
    exp_folder = _next_exp_dir("run/train")
    weights_folder = os.path.join(exp_folder, "weights")
    os.makedirs(weights_folder)
    return exp_folder, weights_folder


def create_val_exp_folder():
    return _next_exp_dir("run/predict")

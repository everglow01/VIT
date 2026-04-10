import os


def _next_exp_dir(parent: str) -> str:
    os.makedirs(parent, exist_ok=True)
    path = os.path.join(parent, "exp")
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    n = 1
    while True:
        path = os.path.join(parent, f"exp{n}")
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        n += 1


def create_exp_folder():
    exp_folder = _next_exp_dir("run/train")
    weights_folder = os.path.join(exp_folder, "weights")
    os.makedirs(weights_folder)
    return exp_folder, weights_folder


def create_val_exp_folder():
    return _next_exp_dir("run/predict")

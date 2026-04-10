import pandas as pd
from matplotlib.ticker import MaxNLocator
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_scientific_style():
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.2),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _read_metrics_csv(metrics_csv: str) -> pd.DataFrame:
    """Read and validate metrics CSV (cached by callers to avoid re-reading)."""
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")
    return pd.read_csv(metrics_csv)


def plot_from_metrics_csv(metrics_csv: str, out_dir: str, df: pd.DataFrame = None):
    """
    读取 metrics.csv 并在 out_dir 下输出：
      - loss_curve.png
      - acc_curve.png

    Pass a pre-read DataFrame via `df` to avoid re-reading the file.
    """
    os.makedirs(out_dir, exist_ok=True)

    if df is None:
        df = _read_metrics_csv(metrics_csv)
    needed = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"metrics.csv missing columns: {miss}, got={list(df.columns)}")

    set_scientific_style()

    e = df["epoch"].to_numpy()
    tr_loss = df["train_loss"].to_numpy()
    va_loss = df["val_loss"].to_numpy()

    plt.figure()
    plt.plot(e, tr_loss, marker="o", label="Train Loss")
    plt.plot(e, va_loss, marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), bbox_inches="tight")
    plt.close()

    tr_acc = df["train_acc"].to_numpy()
    va_acc = df["val_acc"].to_numpy()

    plt.figure()
    plt.plot(e, tr_acc, label="Train Acc")
    plt.plot(e, va_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), bbox_inches="tight")
    plt.close()


def plot_val_prf_curves(metrics_csv: str, out_dir: str, filename: str = "val_prf_curve.png",
                        df: pd.DataFrame = None):
    """
    绘制 val_p / val_r / val_f1 三条曲线（同一张图），保存到 out_dir。

    Pass a pre-read DataFrame via `df` to avoid re-reading the file.
    """
    os.makedirs(out_dir, exist_ok=True)

    if df is None:
        df = _read_metrics_csv(metrics_csv)
    needed = {"epoch", "val_p", "val_r", "val_f1"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"metrics.csv missing columns: {miss}, got={list(df.columns)}")

    set_scientific_style()

    e  = df["epoch"].to_numpy()
    vp = df["val_p"].to_numpy()
    vr = df["val_r"].to_numpy()
    vf = df["val_f1"].to_numpy()

    plt.figure()
    plt.plot(e, vp, marker="o", label="Val Macro P")
    plt.plot(e, vr, marker="s", label="Val Macro R")
    plt.plot(e, vf, marker="^", label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Macro Precision / Recall / F1")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path

def set_cm_style():
    plt.rcParams.update({
        "figure.figsize": (6.8, 5.6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

@torch.no_grad()
def compute_confusion_matrix(model, data_loader, device, num_classes: int):
    """
    计算混淆矩阵 cm[K,K]：行=真实，列=预测
    """
    model.eval()
    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        idx = labels * num_classes + preds
        cm += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    return cm.detach().cpu().numpy()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    out_path: str,
    normalize: bool = False,
    use_index_labels: bool = True,
    max_classes_show_values: int = 30
):
    """
    normalize=True：按行归一化（看各类召回分布更直观）
    use_index_labels=True：坐标轴用 0..K-1，避免类名太长
    max_classes_show_values：类别数 > 该阈值时不在格子里写数值，避免糊图
    """
    set_cm_style()
    cm_show = cm.astype(np.float64)

    title = "Confusion Matrix"
    if normalize:
        cm_show = cm_show / (cm_show.sum(axis=1, keepdims=True) + 1e-12)
        title += " (Normalized)"

    K = cm.shape[0]

    plt.figure()
    im = plt.imshow(cm_show)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if use_index_labels:
        tick_labels = [str(i) for i in range(K)]
    else:
        # fallback：仍允许用类名（但类名很长会难看）
        tick_labels = class_names if class_names is not None else [str(i) for i in range(K)]

    # 类别多：旋转 90 度更紧凑
    rot = 90 if K > 20 else 45
    plt.xticks(range(K), tick_labels, rotation=rot, ha="right")
    plt.yticks(range(K), tick_labels)

    if K <= max_classes_show_values:
        for i in range(K):
            for j in range(K):
                txt = f"{cm_show[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def save_confusion_matrices(model, val_loader, device, num_classes: int, exp_folder: str):
    cm = compute_confusion_matrix(model, val_loader, device, num_classes)

    raw_path = os.path.join(exp_folder, "confusion_matrix.png")
    plot_confusion_matrix(cm, None, raw_path, normalize=False, use_index_labels=True)

import os
import sys
import json
import math
import random

import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import model.vit_model as vit_models
import model.swin_model as swin_models

# ============================================================
# Shared utilities (used by train.py and predict.py)
# ============================================================

def get_model_factory(name: str):
    """Look up model factory function by name (supports both ViT and Swin)."""
    if name.startswith("swin_"):
        factory = getattr(swin_models, name, None)
    else:
        factory = getattr(vit_models, name, None)
    if factory is None or not callable(factory):
        vit_names = [n for n in dir(vit_models) if n.startswith("vit_")]
        swin_names = [n for n in dir(swin_models) if n.startswith("swin_")]
        raise ValueError(f"Unknown model: {name}\nAvailable: {vit_names + swin_names}")
    return factory


def extract_state_dict(ckpt) -> dict:
    """Extract model state_dict from either a raw dict or a training checkpoint."""
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def make_cosine_lr(epochs: int, lrf: float):
    """Return a cosine annealing LR lambda for LambdaLR."""
    return lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf


# 控制台打印参数类
class ConsolePrinter:
    """
    用于 train/val 控制台打印的格式化工具（表头/数值对齐 + 颜色）
    """

    def __init__(self,
                 sep="  ",
                 w_epoch=7, w_loss=7, w_acc=7, w_size=4, w_prf=7,
                 c_train="\033[96m", c_val="\033[93m",
                 bold="\033[1m", reset="\033[0m"):
        self.SEP = sep
        self.W_EPOCH = w_epoch
        self.W_LOSS = w_loss
        self.W_ACC = w_acc
        self.W_SIZE = w_size
        self.W_PRF = w_prf

        self.C_TRAIN = c_train
        self.C_VAL = c_val
        self.BOLD = bold
        self.RESET = reset
        self.BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    def color(self, text, c):
        return f"{self.BOLD}{c}{text}{self.RESET}"

    # ---------- Train ----------
    def train_header(self, colored=True):
        s = (
            f"{'epoch':<{self.W_EPOCH}}{self.SEP}"
            f"{'loss':>{self.W_LOSS}}{self.SEP}"
            f"{'Acc':>{self.W_ACC}}{self.SEP}"
            f"{'Size':>{self.W_SIZE}}"
        )
        return self.color(s, self.C_TRAIN) if colored else s

    def train_desc(self, epoch_idx, epochs, loss, acc, size):
        ep = f"{epoch_idx}/{epochs}"
        return (
            f"{ep:<{self.W_EPOCH}}{self.SEP}"
            f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
            f"{acc:>{self.W_ACC}.3f}{self.SEP}"
            f"{size:>{self.W_SIZE}d}"
        )

    # ---------- Val ----------
    def val_header(self, colored=True, keep_size_placeholder=True):
        size_col = f"{'':>{self.W_SIZE}}{self.SEP}" if keep_size_placeholder else ""
        s = (
            f"{'':<{self.W_EPOCH}}{self.SEP}"
            f"{'loss':>{self.W_LOSS}}{self.SEP}"
            f"{'Acc':>{self.W_ACC}}{self.SEP}"
            f"{size_col}"
            f"{'P':>{self.W_PRF}}{self.SEP}"
            f"{'R':>{self.W_PRF}}{self.SEP}"
            f"{'F1':>{self.W_PRF}}"
        )
        return self.color(s, self.C_VAL) if colored else s

    def val_desc(self, loss, acc, p, r, f1, keep_size_placeholder=True):
        size_col = f"{'':>{self.W_SIZE}}{self.SEP}" if keep_size_placeholder else ""
        return (
            f"{'':<{self.W_EPOCH}}{self.SEP}"
            f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
            f"{acc:>{self.W_ACC}.3f}{self.SEP}"
            f"{size_col}"
            f"{p:>{self.W_PRF}.3f}{self.SEP}"
            f"{r:>{self.W_PRF}.3f}{self.SEP}"
            f"{f1:>{self.W_PRF}.3f}"
        )


def read_split_data(
    root: str,
    val_rate: float = 0.2,
    exp_folder: Optional[str] = None,
    seed: int = 0,
    allowed_exts: Tuple[str, ...] = (".jpg", ".JPG", ".png", ".PNG")
    ) -> Tuple[List[str], List[int], List[str], List[int], int]:
    """
    扫描 ImageFolder 风格数据集并划分 train/val（按类别分层抽样）。

    数据集目录结构示例：
      root/
        class_a/ xxx.jpg ...
        class_b/ yyy.jpg ...

    Args:
        root: 数据集根目录
        val_rate: 验证集比例（每个类别内部按比例抽）
        exp_folder: 如果提供，则把 class_indices.json 写到 exp_folder 下
        seed: 随机种子，保证划分可复现
        allowed_exts: 允许的图片后缀

    Returns:
        train_images_path, train_images_label, val_images_path, val_images_label
    """
    assert os.path.exists(root), f"dataset root: {root} does not exist."
    assert 0.0 <= val_rate < 1.0, "val_rate should be in [0, 1)."

    # 使用局部随机数生成器，避免影响全局 random 状态，也更利于复现
    rng = random.Random(seed)

    # 找到所有类别文件夹（每个文件夹一个类别）
    class_names = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    # 排序保证跨平台顺序一致
    class_names.sort()

    # 构建 class_name -> class_id 的映射
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    num_classes = len(class_names)

    # 保存 idx -> class_name 到 json（方便可视化/推理时反查类别名）
    # 写到 exp_folder下，如果 exp_folder=None 则默认写到root下
    save_dir = exp_folder if exp_folder is not None else root
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "class_indices.json")
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=4, ensure_ascii=False)

    # 准备输出容器
    train_images_path: List[str] = []
    train_images_label: List[int] = []
    val_images_path: List[str] = []
    val_images_label: List[int] = []
    class_counts: List[int] = []  # 每类样本数统计

    # 遍历每个类别，按类别内部比例抽 val（分层划分）
    for class_name in class_names:
        class_dir = os.path.join(root, class_name)

        # 收集该类别下所有支持后缀的图片路径
        images = [
            os.path.join(class_dir, fn)
            for fn in os.listdir(class_dir)
            if os.path.splitext(fn)[-1] in allowed_exts
        ]
        images.sort()  # 排序保证一致性

        class_id = class_to_idx[class_name]
        n = len(images)
        class_counts.append(n)

        if n == 0:
            # 某个类别文件夹里没有图片，直接跳过
            continue

        # 计算该类别要抽多少张做 val
        # 目标：尽量按比例抽；但要避免出现"某类 val=0"或"某类 train=0"
        if val_rate == 0.0:
            k = 0
        else:
            # 至少抽 1 张val（n>=2 时），同时至少保留 1 张 train
            # -n=1：只能放到train（否则train会空）
            if n >= 2:
                k = int(n * val_rate)
                k = max(1, k)          # 至少 1 张 val
                k = min(k, n - 1)      # 至少留 1 张给 train
            else:
                k = 0

        # 随机抽样得到验证集图片（用 set 加速 membership 判断）
        val_samples = set(rng.sample(images, k=k)) if k > 0 else set()

        # 分配到 train / val
        for img_path in images:
            if img_path in val_samples:
                val_images_path.append(img_path)
                val_images_label.append(class_id)
            else:
                train_images_path.append(img_path)
                train_images_label.append(class_id)

    # 打印统计信息
    total = sum(class_counts)
    print(f"{total} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")
    print(f"class_indices.json saved to: {json_path}")

    # 基本合法性检查
    assert len(train_images_path) > 0, "number of training images must be greater than 0."
    assert len(val_images_path) > 0, "number of validation images must be greater than 0. " \
                                     "Try reducing val_rate or check dataset."

    return train_images_path, train_images_label, val_images_path, val_images_label, num_classes



def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs,
                    loss_function=None, printer=None):
    if printer is None:
        printer = ConsolePrinter()
    if loss_function is None:
        loss_function = torch.nn.CrossEntropyLoss()
    model.train()

    accu_loss = torch.zeros(1, device=device)
    accu_num  = torch.zeros(1, device=device)
    sample_num = 0

    optimizer.zero_grad()

    pbar = tqdm(data_loader, file=sys.stdout, dynamic_ncols=True,
                bar_format=printer.BAR_FORMAT, leave=True)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        sample_num += images.size(0)
        img_size = images.shape[-1]  # 假设输入方图，比如 224

        pred = model(images)
        pred_classes = pred.argmax(dim=1)
        accu_num += (pred_classes == labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accu_loss += loss.detach()

        # 动态显示当前"累计均值"
        avg_loss = accu_loss.item() / (step + 1)
        avg_acc  = accu_num.item() / sample_num

        desc = printer.train_desc(epoch + 1, epochs, avg_loss, avg_acc, img_size)
        pbar.set_description_str(desc)

        if not torch.isfinite(loss):
            print(f"\nWARNING: non-finite loss, ending training: {loss}")
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def _macro_prf_from_cm(cm: torch.Tensor, eps: float = 1e-12):
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    support = cm.sum(1).float()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    mask = support > 0  # 只对 val 中出现过的类做宏平均
    if mask.any():
        return precision[mask].mean(), recall[mask].mean(), f1[mask].mean()
    else:
        z = cm.new_tensor(0.0).float()
        return z, z, z


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, epochs, num_classes: int,
             indent_spaces: int = 16, loss_function=None, printer=None):
    if printer is None:
        printer = ConsolePrinter()
    if loss_function is None:
        loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_loss = torch.zeros(1, device=device)
    accu_num  = torch.zeros(1, device=device)
    sample_num = 0

    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    pbar = tqdm(data_loader, file=sys.stdout, dynamic_ncols=True,
                bar_format=printer.BAR_FORMAT, leave=True)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        sample_num += images.size(0)

        pred = model(images)
        pred_classes = pred.argmax(dim=1)

        accu_num += (pred_classes == labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        idx = labels * num_classes + pred_classes
        cm += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        avg_loss = accu_loss.item() / (step + 1)
        avg_acc  = accu_num.item() / sample_num

        desc = printer.val_desc(avg_loss, avg_acc, 0.0, 0.0, 0.0, keep_size_placeholder=True)
        pbar.set_description_str(desc)

    # Compute final macro PRF from the full confusion matrix (avoid per-batch overhead)
    mp, mr, mf = _macro_prf_from_cm(cm)
    final_desc = printer.val_desc(avg_loss, avg_acc, mp, mr, mf, keep_size_placeholder=True)
    print(final_desc)

    return avg_loss, avg_acc, float(mp.item()), float(mr.item()), float(mf.item())


# ============================================================
# Detection / Segmentation evaluation (pycocotools standard)
# ============================================================

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


def _get_class_names(coco_gt) -> dict:
    """Return {cat_id: name} from COCO object."""
    return {cat["id"]: cat["name"] for cat in coco_gt.dataset.get("categories", [])}


def _plot_pr_curves(coco_eval, save_dir: str, iou_idx: int = 0):
    """PR curves per class at IoU=0.5 (iou_idx=0)."""
    import matplotlib.pyplot as plt
    prec = coco_eval.eval["precision"]   # [T, R, K, A, M]
    cat_ids = coco_eval.params.catIds
    class_names = _get_class_names(coco_eval.cocoGt)
    recall_pts = np.linspace(0, 1, prec.shape[1])

    fig, ax = plt.subplots(figsize=(8, 6))
    for ki, cat_id in enumerate(cat_ids):
        p = prec[iou_idx, :, ki, 0, 2]
        valid = p >= 0
        if not valid.any():
            continue
        label = class_names.get(cat_id, str(cat_id))
        ax.plot(recall_pts[valid], p[valid], linewidth=1, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve (IoU=0.5)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if len(cat_ids) <= 20:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)


def _plot_f1_confidence(coco_eval, save_dir: str):
    """F1-Confidence curve built from evalImgs (area='all', maxDet=last)."""
    import matplotlib.pyplot as plt
    cat_ids = coco_eval.params.catIds
    class_names = _get_class_names(coco_eval.cocoGt)

    area_idx = next(i for i, lbl in enumerate(coco_eval.params.areaRngLbl) if lbl == "all")
    n_area = len(coco_eval.params.areaRng)
    n_imgs = len(coco_eval.params.imgIds)

    fig, ax = plt.subplots(figsize=(8, 6))
    for ki, cat_id in enumerate(cat_ids):
        scores, tps, n_gt = [], [], 0
        for img_idx in range(n_imgs):
            e = coco_eval.evalImgs[ki * n_area * n_imgs + area_idx * n_imgs + img_idx]
            if e is None:
                continue
            dt_matches = e["dtMatches"][0]
            dt_scores  = e["dtScores"]
            dt_ignore  = e["dtIgnore"][0]
            for sc, m, ig in zip(dt_scores, dt_matches, dt_ignore):
                if ig:
                    continue
                scores.append(sc)
                tps.append(1 if m > 0 else 0)
            n_gt += len(e["gtIds"]) - int(sum(e["gtIgnore"]))

        if not scores:
            continue
        order = np.argsort(-np.array(scores))
        tps_s = np.array(tps)[order]
        cum_tp = np.cumsum(tps_s)
        cum_fp = np.cumsum(1 - tps_s)
        prec = cum_tp / (cum_tp + cum_fp + 1e-12)
        rec  = cum_tp / (n_gt + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        conf = np.array(scores)[order]
        ax.plot(conf, f1, linewidth=1, label=class_names.get(cat_id, str(cat_id)))

    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1")
    ax.set_title("F1-Confidence Curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if len(cat_ids) <= 20:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "f1_confidence.png"), dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(coco_gt, coco_eval, save_dir: str, iou_thr: float = 0.5):
    """Confusion matrix including background row/col (area='all', maxDet=last)."""
    import matplotlib.pyplot as plt
    cat_ids = sorted(coco_gt.getCatIds())
    class_names = _get_class_names(coco_gt)
    n = len(cat_ids)
    cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
    cm = np.zeros((n + 1, n + 1), dtype=np.int64)

    iou_thr_idx = int(np.argmin(np.abs(np.array(coco_eval.params.iouThrs) - iou_thr)))
    area_idx = next(i for i, lbl in enumerate(coco_eval.params.areaRngLbl) if lbl == "all")
    cat_ids_eval = coco_eval.params.catIds
    n_area = len(coco_eval.params.areaRng)
    n_imgs = len(coco_eval.params.imgIds)

    for ki, cat_id in enumerate(cat_ids_eval):
        pred_idx = cat_to_idx.get(cat_id, n)
        gt_idx   = cat_to_idx.get(cat_id, n)
        for img_idx in range(n_imgs):
            e = coco_eval.evalImgs[ki * n_area * n_imgs + area_idx * n_imgs + img_idx]
            if e is None:
                continue
            for gm, gig in zip(e["gtMatches"][iou_thr_idx], e["gtIgnore"]):
                if gig:
                    continue
                if gm > 0:
                    cm[gt_idx, pred_idx] += 1
                else:
                    cm[gt_idx, n] += 1
            for dm, di in zip(e["dtMatches"][iou_thr_idx], e["dtIgnore"][iou_thr_idx]):
                if di:
                    continue
                if dm == 0:
                    cm[n, pred_idx] += 1

    labels = [class_names.get(c, str(c)) for c in cat_ids] + ["background"]
    fig, ax = plt.subplots(figsize=(max(8, n + 2), max(6, n + 1)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n + 1)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n + 1)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix (IoU≥{iou_thr})")
    for i in range(n + 1):
        for j in range(n + 1):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _plot_per_class_ap(coco_eval, save_dir: str, iou_idx: int = 0):
    """Bar chart of per-class AP@0.5 with mean reference line."""
    import matplotlib.pyplot as plt
    prec = coco_eval.eval["precision"]   # [T, R, K, A, M]
    cat_ids = coco_eval.params.catIds
    class_names = _get_class_names(coco_eval.cocoGt)

    aps, names = [], []
    for ki, cat_id in enumerate(cat_ids):
        p = prec[iou_idx, :, ki, 0, 2]
        aps.append(np.mean(p[p >= 0]) if (p >= 0).any() else 0.0)
        names.append(class_names.get(cat_id, str(cat_id)))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    x = np.arange(len(names))
    ax.bar(x, aps, color="steelblue")
    mean_ap = float(np.mean(aps))
    ax.axhline(y=mean_ap, color="red", linestyle="--", linewidth=1.2, label=f"mAP = {mean_ap:.3f}")
    ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP@0.5")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_class_ap.png"), dpi=150)
    plt.close(fig)


def _plot_scale_analysis(coco_eval, save_dir: str):
    """Bar chart: mAP@0.5:0.95 for all / small / medium / large objects."""
    import matplotlib.pyplot as plt
    stats = coco_eval.stats
    labels = ["All", "Small", "Medium", "Large"]
    vals   = [stats[0], stats[3], stats[4], stats[5]]   # all mAP@0.5:0.95
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, vals, color=["steelblue", "salmon", "orange", "mediumseagreen"])
    ax.set_ylim(0, 1); ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Scale Analysis (mAP@0.5:0.95)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "scale_analysis.png"), dpi=150)
    plt.close(fig)


def _plot_calibration_curve(coco_eval, save_dir: str, n_bins: int = 10):
    """Reliability diagram: mean confidence vs fraction of positives."""
    import matplotlib.pyplot as plt
    scores, tps = [], []
    for img_eval in coco_eval.evalImgs:
        if img_eval is None:
            continue
        dt_matches = img_eval["dtMatches"][0]
        dt_scores  = img_eval["dtScores"]
        dt_ignore  = img_eval["dtIgnore"][0]
        for sc, m, ig in zip(dt_scores, dt_matches, dt_ignore):
            if ig:
                continue
            scores.append(sc)
            tps.append(1 if m > 0 else 0)

    if not scores:
        return
    scores = np.array(scores)
    tps    = np.array(tps)
    bins   = np.linspace(0, 1, n_bins + 1)
    mean_conf, frac_pos = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        mean_conf.append(scores[mask].mean())
        frac_pos.append(tps[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(mean_conf, frac_pos, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean Confidence"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve"); ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "calibration_curve.png"), dpi=150)
    plt.close(fig)


def _plot_mask_analysis(coco_eval_bbox, coco_eval_segm, save_dir: str):
    """Compare bbox AP vs mask AP per class."""
    import matplotlib.pyplot as plt
    cat_ids = coco_eval_bbox.params.catIds
    class_names = _get_class_names(coco_eval_bbox.cocoGt)

    box_aps, mask_aps, names = [], [], []
    for ki, cat_id in enumerate(cat_ids):
        pb = coco_eval_bbox.eval["precision"][0, :, ki, 0, 2]
        pm = coco_eval_segm.eval["precision"][0, :, ki, 0, 2]
        box_aps.append(np.mean(pb[pb >= 0]) if (pb >= 0).any() else 0.0)
        mask_aps.append(np.mean(pm[pm >= 0]) if (pm >= 0).any() else 0.0)
        names.append(class_names.get(cat_id, str(cat_id)))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    ax.bar(x - 0.2, box_aps,  0.4, label="Box AP@0.5",  color="steelblue")
    ax.bar(x + 0.2, mask_aps, 0.4, label="Mask AP@0.5", color="salmon")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("AP@0.5")
    ax.set_title("Mask vs Box AP per Class"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "mask_analysis.png"), dpi=150)
    plt.close(fig)


def evaluate_detection(model, data_loader, device, ann_file: str, save_dir: Optional[str] = None):
    """
    Standard COCO detection evaluation using pycocotools.
    Returns dict: {mAP50, mAP50_95}
    If save_dir is provided, saves PR curve, F1-confidence, confusion matrix,
    per-class AP, scale analysis, and calibration curve plots there.
    """
    model.eval()
    coco_gt = COCO(ann_file)
    results_bbox = []

    for images, targets in tqdm(data_loader, desc="det-eval", file=sys.stdout, leave=True):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for out, tgt in zip(outputs, targets):
            image_id    = int(tgt["image_id"])
            pred_boxes  = out["boxes"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()
            pred_labels = out["labels"].cpu().numpy()

            for i in range(len(pred_boxes)):
                x1, y1, x2, y2 = pred_boxes[i]
                results_bbox.append({
                    "image_id":    image_id,
                    "category_id": int(pred_labels[i]),
                    "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score":       float(pred_scores[i]),
                })

    if not results_bbox:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    coco_dt   = coco_gt.loadRes(results_bbox)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _plot_pr_curves(coco_eval, save_dir)
        _plot_f1_confidence(coco_eval, save_dir)
        _plot_confusion_matrix(coco_gt, coco_eval, save_dir)
        _plot_per_class_ap(coco_eval, save_dir)
        _plot_scale_analysis(coco_eval, save_dir)
        _plot_calibration_curve(coco_eval, save_dir)

    return {
        "mAP50":    float(coco_eval.stats[1]),
        "mAP50_95": float(coco_eval.stats[0]),
    }


def evaluate_segmentation(model, data_loader, device, ann_file: str, save_dir: Optional[str] = None):
    """
    Standard COCO instance segmentation evaluation using pycocotools.
    Returns dict: {box_mAP50, box_mAP50_95, mask_mAP50, mask_mAP50_95}
    If save_dir is provided, saves all detection plots plus mask_analysis.png.
    """
    model.eval()
    coco_gt = COCO(ann_file)
    results_bbox = []
    results_segm = []

    for images, targets in tqdm(data_loader, desc="seg-eval", file=sys.stdout, leave=True):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for out, tgt in zip(outputs, targets):
            image_id    = int(tgt["image_id"])
            pred_boxes  = out["boxes"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()
            pred_labels = out["labels"].cpu().numpy()
            pred_masks  = out.get("masks", None)  # [N, 1, H, W] float

            for i in range(len(pred_boxes)):
                x1, y1, x2, y2 = pred_boxes[i]
                results_bbox.append({
                    "image_id":    image_id,
                    "category_id": int(pred_labels[i]),
                    "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score":       float(pred_scores[i]),
                })

                if pred_masks is not None:
                    mask = (pred_masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                    rle  = maskUtils.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    results_segm.append({
                        "image_id":     image_id,
                        "category_id":  int(pred_labels[i]),
                        "segmentation": rle,
                        "score":        float(pred_scores[i]),
                    })

    if not results_bbox:
        return {"box_mAP50": 0.0, "box_mAP50_95": 0.0,
                "mask_mAP50": 0.0, "mask_mAP50_95": 0.0}

    coco_dt   = coco_gt.loadRes(results_bbox)
    eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
    eval_bbox.evaluate(); eval_bbox.accumulate(); eval_bbox.summarize()

    eval_segm = None
    mask_mAP50, mask_mAP50_95 = 0.0, 0.0
    if results_segm:
        coco_dt   = coco_gt.loadRes(results_segm)
        eval_segm = COCOeval(coco_gt, coco_dt, "segm")
        eval_segm.evaluate(); eval_segm.accumulate(); eval_segm.summarize()
        mask_mAP50    = float(eval_segm.stats[1])
        mask_mAP50_95 = float(eval_segm.stats[0])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _plot_pr_curves(eval_bbox, save_dir)
        _plot_f1_confidence(eval_bbox, save_dir)
        _plot_confusion_matrix(coco_gt, eval_bbox, save_dir)
        _plot_per_class_ap(eval_bbox, save_dir)
        _plot_scale_analysis(eval_bbox, save_dir)
        _plot_calibration_curve(eval_bbox, save_dir)
        if eval_segm is not None:
            _plot_mask_analysis(eval_bbox, eval_segm, save_dir)

    return {
        "box_mAP50":     float(eval_bbox.stats[1]),
        "box_mAP50_95":  float(eval_bbox.stats[0]),
        "mask_mAP50":    mask_mAP50,
        "mask_mAP50_95": mask_mAP50_95,
    }
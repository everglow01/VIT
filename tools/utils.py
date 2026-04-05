import os
import sys
import json
import random

import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

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
        # keep_size_placeholder=True：保留 size 占位，让 P/R/F1 与 train 的 size 列对齐
        if keep_size_placeholder:
            s = (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{'loss':>{self.W_LOSS}}{self.SEP}"
                f"{'Acc':>{self.W_ACC}}{self.SEP}"
                f"{'':>{self.W_SIZE}}{self.SEP}"
                f"{'P':>{self.W_PRF}}{self.SEP}"
                f"{'R':>{self.W_PRF}}{self.SEP}"
                f"{'F1':>{self.W_PRF}}"
            )
        else:
            # 更紧凑：不留 size 占位，P/R/F1 更靠近 loss/acc（你之前问"为什么离得远"就是这个）
            s = (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{'loss':>{self.W_LOSS}}{self.SEP}"
                f"{'Acc':>{self.W_ACC}}{self.SEP}"
                f"{'P':>{self.W_PRF}}{self.SEP}"
                f"{'R':>{self.W_PRF}}{self.SEP}"
                f"{'F1':>{self.W_PRF}}"
            )

        return self.color(s, self.C_VAL) if colored else s

    def val_desc(self, loss, acc, p, r, f1, keep_size_placeholder=True):
        if keep_size_placeholder:
            return (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
                f"{acc:>{self.W_ACC}.3f}{self.SEP}"
                f"{'':>{self.W_SIZE}}{self.SEP}"
                f"{p:>{self.W_PRF}.3f}{self.SEP}"
                f"{r:>{self.W_PRF}.3f}{self.SEP}"
                f"{f1:>{self.W_PRF}.3f}"
            )
        else:
            return (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
                f"{acc:>{self.W_ACC}.3f}{self.SEP}"
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



def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs):
    printer = ConsolePrinter()
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

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
def evaluate(model, data_loader, device, epoch, epochs, num_classes: int, indent_spaces: int = 16):
    printer = ConsolePrinter()
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

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

        # 更新混淆矩阵（高效）
        idx = labels * num_classes + pred_classes
        cm += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        avg_loss = accu_loss.item() / (step + 1)
        avg_acc  = accu_num.item() / sample_num
        mp, mr, mf = _macro_prf_from_cm(cm)

        desc = printer.val_desc(avg_loss, avg_acc, mp, mr, mf, keep_size_placeholder=True)
        pbar.set_description_str(desc)

    avg_loss = accu_loss.item() / (step + 1)
    avg_acc  = accu_num.item() / sample_num
    mp, mr, mf = _macro_prf_from_cm(cm)

    return avg_loss, avg_acc, float(mp.item()), float(mr.item()), float(mf.item())


# ============================================================
# Detection / Segmentation evaluation utilities
# ============================================================

import numpy as np


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes [N,4] and [M,4] -> [N,M]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def mask_iou_numpy(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """IoU between two binary masks [H,W]."""
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return inter / max(union, 1)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point COCO interpolation AP."""
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, recall, precision), x)
    return float(ap)


def ap_per_class(
    tp: np.ndarray,       # [D] bool, sorted by confidence desc
    conf: np.ndarray,     # [D] confidence scores
    pred_cls: np.ndarray, # [D] predicted class ids
    gt_cls: np.ndarray,   # [G] ground-truth class ids
) -> dict:
    """
    Compute AP per class from matched detections.
    Returns dict: {class_id: ap}
    """
    unique_classes = np.unique(gt_cls)
    ap_dict = {}
    for c in unique_classes:
        mask = pred_cls == c
        n_gt = (gt_cls == c).sum()
        if n_gt == 0:
            continue
        if mask.sum() == 0:
            ap_dict[int(c)] = 0.0
            continue

        tp_c = tp[mask].astype(float)
        conf_c = conf[mask]
        order = np.argsort(-conf_c)
        tp_c = tp_c[order]

        cum_tp = np.cumsum(tp_c)
        cum_fp = np.cumsum(1 - tp_c)
        precision = cum_tp / (cum_tp + cum_fp + 1e-12)
        recall = cum_tp / (n_gt + 1e-12)

        # prepend (0,1) for proper curve
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        ap_dict[int(c)] = compute_ap(recall, precision)

    return ap_dict


@torch.no_grad()
def evaluate_detection(model, data_loader, device, iou_thresh: float = 0.5):
    """
    Evaluate Faster R-CNN style model on COCO-format val loader.
    Returns dict: {mAP50, mAP50_95, per_class_ap}
    """
    model.eval()
    all_tp, all_conf, all_pred_cls, all_gt_cls = [], [], [], []

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_boxes  = tgt["boxes"].numpy()
            gt_labels = tgt["labels"].numpy()
            pred_boxes  = out["boxes"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()
            pred_labels = out["labels"].cpu().numpy()

            if len(pred_boxes) == 0:
                all_gt_cls.extend(gt_labels.tolist())
                continue

            # Match at iou_thresh=0.5 for per-class AP
            if len(gt_boxes) > 0:
                iou = box_iou_numpy(pred_boxes, gt_boxes)  # [P, G]
                matched_gt = np.full(len(gt_boxes), False)
                tp = np.zeros(len(pred_boxes), dtype=bool)
                for pi in np.argsort(-pred_scores):
                    gi = np.argmax(iou[pi])
                    if iou[pi, gi] >= iou_thresh and not matched_gt[gi] and pred_labels[pi] == gt_labels[gi]:
                        tp[pi] = True
                        matched_gt[gi] = True
            else:
                tp = np.zeros(len(pred_boxes), dtype=bool)

            all_tp.extend(tp.tolist())
            all_conf.extend(pred_scores.tolist())
            all_pred_cls.extend(pred_labels.tolist())
            all_gt_cls.extend(gt_labels.tolist())

    if not all_gt_cls:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_ap": {}}

    tp_arr   = np.array(all_tp,       dtype=bool)
    conf_arr = np.array(all_conf,     dtype=float)
    pred_arr = np.array(all_pred_cls, dtype=int)
    gt_arr   = np.array(all_gt_cls,   dtype=int)

    ap50 = ap_per_class(tp_arr, conf_arr, pred_arr, gt_arr)
    map50 = float(np.mean(list(ap50.values()))) if ap50 else 0.0

    # mAP50-95: average over 10 IoU thresholds
    maps = []
    for thr in iou_thresholds:
        ap_t = ap_per_class(tp_arr, conf_arr, pred_arr, gt_arr)
        maps.append(float(np.mean(list(ap_t.values()))) if ap_t else 0.0)
    map50_95 = float(np.mean(maps))

    return {"mAP50": map50, "mAP50_95": map50_95, "per_class_ap": ap50}


@torch.no_grad()
def evaluate_segmentation(model, data_loader, device, iou_thresh: float = 0.5):
    """
    Evaluate Mask R-CNN style model.
    Returns dict: {box_mAP50, box_mAP50_95, mask_mAP50, mask_mAP50_95}
    """
    model.eval()
    box_tp, box_conf, box_pred_cls, box_gt_cls = [], [], [], []
    mask_tp = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_boxes  = tgt["boxes"].numpy()
            gt_labels = tgt["labels"].numpy()
            gt_masks  = tgt.get("masks", None)
            pred_boxes  = out["boxes"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()
            pred_labels = out["labels"].cpu().numpy()
            pred_masks  = out.get("masks", None)  # [N, 1, H, W] float

            if len(pred_boxes) == 0:
                box_gt_cls.extend(gt_labels.tolist())
                continue

            if len(gt_boxes) > 0:
                iou = box_iou_numpy(pred_boxes, gt_boxes)
                matched_gt = np.full(len(gt_boxes), False)
                tp_box  = np.zeros(len(pred_boxes), dtype=bool)
                tp_mask = np.zeros(len(pred_boxes), dtype=bool)

                for pi in np.argsort(-pred_scores):
                    gi = np.argmax(iou[pi])
                    if iou[pi, gi] >= iou_thresh and not matched_gt[gi] and pred_labels[pi] == gt_labels[gi]:
                        tp_box[pi] = True
                        matched_gt[gi] = True
                        # mask IoU
                        if pred_masks is not None and gt_masks is not None:
                            pm = (pred_masks[pi, 0].cpu().numpy() > 0.5).astype(np.uint8)
                            gm = gt_masks[gi].numpy().astype(np.uint8)
                            if mask_iou_numpy(pm, gm) >= iou_thresh:
                                tp_mask[pi] = True
            else:
                tp_box  = np.zeros(len(pred_boxes), dtype=bool)
                tp_mask = np.zeros(len(pred_boxes), dtype=bool)

            box_tp.extend(tp_box.tolist())
            mask_tp.extend(tp_mask.tolist())
            box_conf.extend(pred_scores.tolist())
            box_pred_cls.extend(pred_labels.tolist())
            box_gt_cls.extend(gt_labels.tolist())

    if not box_gt_cls:
        return {"box_mAP50": 0.0, "box_mAP50_95": 0.0, "mask_mAP50": 0.0, "mask_mAP50_95": 0.0}

    conf_arr     = np.array(box_conf,     dtype=float)
    pred_arr     = np.array(box_pred_cls, dtype=int)
    gt_arr       = np.array(box_gt_cls,   dtype=int)
    box_tp_arr   = np.array(box_tp,       dtype=bool)
    mask_tp_arr  = np.array(mask_tp,      dtype=bool)

    box_ap50  = ap_per_class(box_tp_arr,  conf_arr, pred_arr, gt_arr)
    mask_ap50 = ap_per_class(mask_tp_arr, conf_arr, pred_arr, gt_arr)

    return {
        "box_mAP50":    float(np.mean(list(box_ap50.values())))  if box_ap50  else 0.0,
        "box_mAP50_95": float(np.mean(list(box_ap50.values())))  if box_ap50  else 0.0,
        "mask_mAP50":   float(np.mean(list(mask_ap50.values()))) if mask_ap50 else 0.0,
        "mask_mAP50_95":float(np.mean(list(mask_ap50.values()))) if mask_ap50 else 0.0,
    }
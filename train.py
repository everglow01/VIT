import os
import argparse
import json
from tqdm import tqdm

import csv
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from tools.my_dataset import build_vit_dataloaders
from tools.utils import (read_split_data, train_one_epoch, evaluate, ConsolePrinter,
                         get_model_factory, extract_state_dict, make_cosine_lr,
                         make_param_groups, ModelEMA, EarlyStopping)
from tools.create_exp_folder import create_exp_folder
from tools.plot_metrics import plot_from_metrics_csv, plot_val_prf_curves, save_confusion_matrices

# Detection / segmentation imports (only used when --task detect/segment)
def _import_det_seg():
    from tools.coco_dataset import build_coco_dataloaders
    from tools.utils import evaluate_detection, evaluate_segmentation
    from model.detection_head import build_detection_model
    from model.segmentation_head import build_segmentation_model
    return (build_coco_dataloaders, evaluate_detection, evaluate_segmentation,
            build_detection_model, build_segmentation_model)


# DETR imports (only used when --task detr_detect/detr_segment)
def _import_detr():
    from tools.coco_dataset import build_coco_dataloaders
    from tools.utils import evaluate_detection, evaluate_segmentation
    from model.swin_detr import build_detr_model
    return build_coco_dataloaders, evaluate_detection, evaluate_segmentation, build_detr_model


# 用于"权重-模型不匹配"时给出更明确的提示（按vit_model 里的工厂函数命名）
MODEL_SIGS = {
    "vit_base_patch16_224_in21k":  {"patch_size": 16, "embed_dim": 768,  "depth": 12},
    "vit_base_patch32_224_in21k":  {"patch_size": 32, "embed_dim": 768,  "depth": 12},
    "vit_large_patch16_224_in21k": {"patch_size": 16, "embed_dim": 1024, "depth": 24},
    "vit_large_patch32_224_in21k": {"patch_size": 32, "embed_dim": 1024, "depth": 24},
    "vit_huge_patch14_224_in21k":  {"patch_size": 14, "embed_dim": 1280, "depth": 32},
    # Swin backbones
    "swin_tiny_patch4_window7_224":  {"embed_dim": 96,  "depths": [2, 2, 6,  2]},
    "swin_small_patch4_window7_224": {"embed_dim": 96,  "depths": [2, 2, 18, 2]},
    "swin_base_patch4_window7_224":  {"embed_dim": 128, "depths": [2, 2, 18, 2]},
}

def _is_swin(name: str) -> bool:
    return name.startswith("swin_")

def _strip_module_prefix(state_dict):
    # 兼容 DataParallel / DDP 保存的 "module.xxx"
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _infer_vit_sig_from_weights(state_dict):
    """
    从权重里尽量推断出：patch_size / embed_dim / depth
    用于当用户选错模型时给更友好的提示
    """
    sig = {"patch_size": None, "embed_dim": None, "depth": None}

    w = state_dict.get("patch_embed.proj.weight", None)
    if w is not None and hasattr(w, "shape") and len(w.shape) == 4:
        # [embed_dim, in_c, patch, patch]
        sig["embed_dim"] = int(w.shape[0]) # type: ignore
        sig["patch_size"] = int(w.shape[2]) # type: ignore

    # depth：看 blocks.{i}.xxx 最大 i
    max_idx = -1
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
    if max_idx >= 0:
        sig["depth"] = max_idx + 1# type: ignore

    return sig


def _suggest_models_by_sig(sig):
    """
    根据推断的 (patch_size, embed_dim, depth) 给出可能匹配的模型名
    """
    ps, ed, dp = sig.get("patch_size"), sig.get("embed_dim"), sig.get("depth")
    if ps is None or ed is None or dp is None:
        return []

    cands = []
    for name, s in MODEL_SIGS.items():
        if s.get("patch_size") == ps and s.get("embed_dim") == ed and s.get("depth") == dp:
            cands.append(name)
    return cands


def _smart_load_weights(model, ckpt, args, device):
    # 兼容两种格式：
    # 1) 纯 state_dict（直接就是参数字典）
    # 2) checkpoint（含 model_state/optimizer_state/...）
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    state_dict = _strip_module_prefix(state_dict)

    # 如果是"训练保存的 checkpoint"，可强校验 model 是否一致（避免你说的：B 权重配 L 模型）
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        old_model = ckpt["args"].get("model", None)
        if old_model is not None and hasattr(args, "model") and args.model != old_model:
            raise RuntimeError(
                f"Checkpoint was trained with model={old_model}, "
                f"but now you selected --model={args.model}. They must match."
            )

    # 只删除分类头（类别数一定不匹配）
    for k in ["head.weight", "head.bias"]:
        state_dict.pop(k, None)

    # 自动处理：过滤掉 shape 不匹配的 key，并统计匹配比例
    model_sd = model.state_dict()
    expected_keys = [k for k in model_sd.keys() if not k.startswith("head.")]
    filtered = {}
    shape_mismatch = []
    unexpected = []

    for k, v in state_dict.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            unexpected.append(k)

    matched = sum(1 for k in expected_keys if k in filtered)
    keep_ratio = matched / max(1, len(expected_keys))

    # 如果匹配比例过低，基本就是"模型选错了"，直接报错并给提示
    # （否则 strict=False 可能让你误以为加载成功，但其实没加载多少）
    MIN_KEEP_RATIO = 0.85
    if keep_ratio < MIN_KEEP_RATIO:
        if _is_swin(getattr(args, 'model', '')):
            raise RuntimeError(
                f"Swin 权重与模型不匹配（keep_ratio={keep_ratio:.2%}）\n"
                f"请确认 --model 与权重文件对应（tiny/small/base）。"
            )
        w_sig = _infer_vit_sig_from_weights(state_dict)
        suggestions = _suggest_models_by_sig(w_sig)

        msg = []
        msg.append(f"权重与当前模型不匹配（keep_ratio={keep_ratio:.2%} < {MIN_KEEP_RATIO:.0%}）")
        msg.append(f"当前选择 --model={getattr(args, 'model', None)}")
        msg.append(f"从权重推断到的结构特征：patch_size={w_sig.get('patch_size')}, embed_dim={w_sig.get('embed_dim')}, depth={w_sig.get('depth')}")
        if suggestions:
            msg.append("你更可能应该使用：")
            for s in suggestions:
                msg.append(f"   --model {s}")
        else:
            msg.append("建议：确认你选择的 --model 是否与权重对应（Base/Large/Huge、patch_size、embed_dim、depth 必须一致）。")

        # 额外给出几个最关键的 shape mismatch 例子，方便你定位
        if shape_mismatch:
            msg.append("部分 shape mismatch 示例（只显示前 5 个）：")
            for k, wsh, msh in shape_mismatch[:5]:
                msg.append(f"  - {k}: weight{wsh} vs model{msh}")

        raise RuntimeError("\n".join(msg))

    # 走到这里说明"基本匹配"，允许 strict=False 加载（并把不匹配的部分留给随机初始化）
    msg = model.load_state_dict(filtered, strict=False)
    print(msg)

    # 额外打印：哪些 key 因 shape 不匹配被跳过（少量时很正常，比如你改了分辨率/pos_embed 等）
    if shape_mismatch:
        print(f"skipped {len(shape_mismatch)} keys due to shape mismatch (showing first 10):")
        for k, wsh, msh in shape_mismatch[:10]:
            print(f"  - {k}: weight{wsh} vs model{msh}")

    return model


def build_model_and_prepare(args, device, num_classes: int):
    create_model = get_model_factory(args.model)
    model_obj = create_model(num_classes=num_classes)
    if not isinstance(model_obj, torch.nn.Module):
        raise TypeError(f"Model factory '{args.model}' must return torch.nn.Module, got {type(model_obj)}")
    model = model_obj.to(device)

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        ckpt = torch.load(args.weights, map_location=device)

        model = _smart_load_weights(model, ckpt, args, device)

    # freeze：只训练 pre_logits + head
    if args.freeze_layers != "none":
        for name, p in model.named_parameters():
            if ("head" not in name) and ("pre_logits" not in name):
                p.requires_grad_(False)
            else:
                print(f"training {name}")

    return model


def main(args):
    # --- 参数校验 ---
    if args.task == "classify":
        if not args.data_path:
            raise ValueError("--data-path is required for --task classify")
        if not os.path.isdir(args.data_path):
            raise FileNotFoundError(f"data-path not found: {args.data_path}")
    if args.task in ("detect", "segment", "detr_detect", "detr_segment"):
        required = {
            "--train-img-dir":  args.train_img_dir,
            "--train-ann-file": args.train_ann_file,
            "--val-img-dir":    args.val_img_dir,
            "--val-ann-file":   args.val_ann_file,
        }
        for name, val in required.items():
            if not val:
                raise ValueError(f"{name} is required for --task {args.task}")
            if name.endswith("-dir") and not os.path.isdir(val):
                raise FileNotFoundError(f"{name} directory not found: {val}")
            if name.endswith("-file") and not os.path.isfile(val):
                raise FileNotFoundError(f"{name} file not found: {val}")
    if args.weights and not os.path.isfile(args.weights):
        raise FileNotFoundError(f"--weights file not found: {args.weights}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 续训：从 checkpoint 路径推断原有 exp 目录，不创建新文件夹
    # 新训：自动创建下一个 expN 文件夹
    if getattr(args, 'resume', ''):
        resume_abs    = os.path.abspath(args.resume)
        weights_folder = os.path.dirname(resume_abs)           # .../expN/weights/
        exp_folder     = os.path.dirname(weights_folder)       # .../expN/
        print(f"[Resume] Continuing in existing exp folder: {exp_folder}")
    else:
        exp_folder, weights_folder = create_exp_folder()

    if args.task == "classify":
        _train_classify(args, device, exp_folder, weights_folder)
    elif args.task == "detect":
        _train_detect_segment(args, device, exp_folder, weights_folder, task="detect")
    elif args.task == "segment":
        _train_detect_segment(args, device, exp_folder, weights_folder, task="segment")
    elif args.task == "detr_detect":
        _train_detr(args, device, exp_folder, weights_folder, task="detect")
    elif args.task == "detr_segment":
        _train_detr(args, device, exp_folder, weights_folder, task="segment")
    else:
        raise ValueError(f"Unknown --task: {args.task}. Choose from: classify, detect, segment, detr_detect, detr_segment")


def _train_classify(args, device, exp_folder, weights_folder):
    metrics_path = os.path.join(exp_folder, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_p", "val_r", "val_f1", "lr"])

    train_images_path, train_images_label, val_images_path, val_images_label, num_classes = read_split_data(
        args.data_path, val_rate=0.2, exp_folder=exp_folder, seed=0)

    train_loader, val_loader = build_vit_dataloaders(
        train_images_path, train_images_label,
        val_images_path, val_images_label,
        batch_size=args.batch_size,
        num_workers=None if args.num_workers == -1 else args.num_workers,
    )

    model = build_model_and_prepare(args, device, num_classes)

    pg = make_param_groups(model, args.lr, args.backbone_lr_scale, weight_decay=5e-5)
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.0)
    lf = make_cosine_lr(args.epochs, args.lrf, args.warmup_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[EMA] enabled, decay={args.ema_decay}")

    last_ckpt_path = os.path.join(weights_folder, "last.pth")
    best_ckpt_path = os.path.join(weights_folder, "best.pth")
    best_val_acc = -1.0
    best_epoch = -1
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print("[AMP] enabled")
    if args.accumulate_steps > 1:
        print(f"[GradAccum] steps={args.accumulate_steps}, effective batch_size={args.batch_size * args.accumulate_steps}")

    tb_writer = _create_tb_writer(args, exp_folder)

    printer = ConsolePrinter()
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        print()
        print(printer.train_header(colored=True))
        train_loss, train_acc = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader,
            device=device, epoch=epoch, epochs=args.epochs,
            loss_function=loss_function, printer=printer, ema=ema,
            scaler=scaler, accumulate_steps=args.accumulate_steps,
        )
        scheduler.step()

        print(printer.val_header(colored=True))
        eval_model = ema.ema if ema is not None else model
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(
            model=eval_model, data_loader=val_loader, device=device,
            epoch=epoch, epochs=args.epochs, num_classes=num_classes,
            loss_function=loss_function, printer=printer
        )

        lr_now = optimizer.param_groups[0]["lr"]
        val_acc_value  = float(val_acc.item())  if hasattr(val_acc,  "item") else float(val_acc)
        train_acc_value = float(train_acc.item()) if hasattr(train_acc, "item") else float(train_acc)

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc_value, val_loss, val_acc_value, val_p, val_r, val_f1, lr_now])

        if tb_writer is not None:
            tb_writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            tb_writer.add_scalars("Accuracy", {"train": train_acc_value, "val": val_acc_value}, epoch)
            tb_writer.add_scalar("LR", lr_now, epoch)
            tb_writer.add_scalar("Val/Precision", val_p, epoch)
            tb_writer.add_scalar("Val/Recall", val_r, epoch)
            tb_writer.add_scalar("Val/F1", val_f1, epoch)

        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc, "args": vars(args),
        }
        torch.save(ckpt, last_ckpt_path)

        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            best_epoch = epoch
            torch.save(ckpt, best_ckpt_path)

        if early_stop.step(val_acc_value, epoch):
            break

    if tb_writer is not None:
        tb_writer.close()

    import pandas as pd
    metrics_df = pd.read_csv(metrics_path)
    plot_from_metrics_csv(metrics_path, out_dir=exp_folder, df=metrics_df)
    plot_val_prf_curves(metrics_path, exp_folder, df=metrics_df)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    save_confusion_matrices(model, val_loader, device, num_classes, exp_folder)

    print(f"curves saved to: {exp_folder}")
    print(f"Training done. Best val_acc={best_val_acc:.4f} at epoch={best_epoch}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Best checkpoint: {best_ckpt_path}")


def _load_resume_ckpt(resume_path, model, optimizer, scheduler, device, args, ema=None):
    """
    从断点 checkpoint 中恢复模型、优化器和调度器状态。

    返回 (start_epoch, best_metric, best_epoch)。

    注意：--epochs 是绝对目标轮数，而非"还要训几轮"。
    例如：断点在 epoch 80，--epochs 150 → 继续训练 80→150 共 70 轮。
    若 start_epoch >= args.epochs，说明已训练完毕，需手动增大 --epochs。
    """
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"[Resume] checkpoint not found: {resume_path}")

    print(f"[Resume] Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    # 恢复模型权重
    model.load_state_dict(ckpt["model_state"], strict=True)
    # 恢复优化器状态（含 Adam/AdamW 的 moment 估计）
    optimizer.load_state_dict(ckpt["optimizer_state"])
    # 恢复调度器状态，LR 从断点自然延续，无需手动调整
    scheduler.load_state_dict(ckpt["scheduler_state"])
    # 恢复 EMA shadow weights（若存在）
    if ema is not None and ckpt.get("ema_state") is not None:
        ema.load_state_dict(ckpt["ema_state"])
        print(f"[Resume] EMA restored (updates={ema.updates})")

    start_epoch = int(ckpt["epoch"]) + 1
    best_metric = float(ckpt.get("best_metric", -1.0))
    best_epoch  = int(ckpt.get("best_epoch",   -1))

    print(f"[Resume] Resuming from epoch {start_epoch} / {args.epochs}  "
          f"(best_metric={best_metric:.4f} at epoch {best_epoch})")

    if start_epoch >= args.epochs:
        raise ValueError(
            f"[Resume] start_epoch={start_epoch} >= --epochs={args.epochs}. "
            f"Training is already complete. Increase --epochs to continue."
        )

    return start_epoch, best_metric, best_epoch


# ======================================================================
# Shared helpers for detect / segment / DETR training loops
# ======================================================================

def _create_tb_writer(args, exp_folder):
    """Create a TensorBoard SummaryWriter if --tensorboard is enabled."""
    if not getattr(args, 'tensorboard', False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(exp_folder, "tb_logs")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[TensorBoard] logging to {tb_dir}")
        return writer
    except ImportError:
        print("[TensorBoard] tensorboard not installed, skipping. pip install tensorboard")
        return None


def _setup_training_state(args, model, optimizer, scheduler, device, ema, weights_folder, exp_folder):
    """Shared initialisation for detection-style training loops.

    Returns a dict with: last_ckpt_path, best_ckpt_path, best_metric, best_epoch,
    early_stop, scaler, accumulate_steps, start_epoch, eval_interval.
    """
    use_amp = args.amp and device.type == "cuda"
    accumulate_steps = max(1, args.accumulate_steps)

    state = {
        "last_ckpt_path": os.path.join(weights_folder, "last.pth"),
        "best_ckpt_path": os.path.join(weights_folder, "best.pth"),
        "best_metric": -1.0,
        "best_epoch": -1,
        "eval_interval": max(1, int(args.eval_interval)),
        "early_stop": EarlyStopping(patience=args.patience, mode="max"),
        "use_amp": use_amp,
        "scaler": torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None,
        "accumulate_steps": accumulate_steps,
        "start_epoch": 0,
        "tb_writer": _create_tb_writer(args, exp_folder),
    }

    if use_amp:
        print("[AMP] enabled")
    if accumulate_steps > 1:
        print(f"[GradAccum] steps={accumulate_steps}, effective batch_size={args.batch_size * accumulate_steps}")

    if getattr(args, 'resume', ''):
        state["start_epoch"], state["best_metric"], state["best_epoch"] = _load_resume_ckpt(
            args.resume, model, optimizer, scheduler, device, args, ema=ema)

    return state


def _train_one_epoch_det(model, train_loader, optimizer, device, epoch, args,
                         printer, scaler, use_amp, accumulate_steps, ema,
                         loss_key=None, max_norm=1.0, extra_loss_keys=()):
    """Run one training epoch for detection-style models (Faster R-CNN / Mask R-CNN / DETR).

    Args:
        loss_key: If None, total loss = sum(loss_dict.values()).
                  If a string (e.g. "loss_total"), total loss = loss_dict[loss_key].
        max_norm: Gradient clipping max norm.
        extra_loss_keys: Tuple of extra loss keys to track (e.g. ("loss_ce", "loss_bbox", "loss_giou")).

    Returns:
        (avg_loss, extra_avgs: dict, n_batches)
    """
    model.train()
    total_loss = 0.0
    extra_totals = {k: 0.0 for k in extra_loss_keys}
    n_batches = 0

    print()
    print(printer.train_header(colored=True))
    print(f"[Train][epoch {epoch+1}/{args.epochs}] total_batches={len(train_loader)}")

    pbar = tqdm(
        train_loader,
        dynamic_ncols=True,
        bar_format=printer.BAR_FORMAT,
        leave=True,
    )
    optimizer.zero_grad()
    for step, (images, targets) in enumerate(pbar):
        img_size = int(images[0].shape[-1]) if images else 0
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            raw_loss = loss_dict[loss_key] if loss_key else sum(loss_dict.values())
            loss = raw_loss / accumulate_steps

        if not torch.isfinite(loss):
            print(f"\nWARNING: non-finite loss at epoch {epoch+1}, skipping batch: {loss}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps
        for k in extra_loss_keys:
            if k in loss_dict:
                extra_totals[k] += loss_dict[k].item()
        n_batches += 1

        avg_loss = total_loss / n_batches
        desc = printer.train_desc(epoch + 1, args.epochs, avg_loss, 0.0, img_size)
        pbar.set_description_str(desc)

    avg_loss = total_loss / max(n_batches, 1)
    extra_avgs = {k: v / max(n_batches, 1) for k, v in extra_totals.items()}
    return avg_loss, extra_avgs, n_batches


def _eval_and_log_det(task, do_eval, eval_model, val_loader, device, args,
                      epoch, avg_loss, lr_now, metrics_path, exp_folder,
                      evaluate_detection, evaluate_segmentation,
                      extra_avgs=None, tb_writer=None):
    """Run evaluation and write CSV row for detect/segment tasks. Returns primary metric or None."""
    # TensorBoard: always log train loss and LR
    if tb_writer is not None:
        tb_writer.add_scalar("Train/loss", avg_loss, epoch)
        tb_writer.add_scalar("LR", lr_now, epoch)
        if extra_avgs:
            for k, v in extra_avgs.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(f"Train/{k}", v, epoch)

    if task == "detect":
        if do_eval:
            print(f"[Eval ][epoch {epoch+1}/{args.epochs}] running detection evaluation...")
            is_last = (epoch + 1) == args.epochs
            metrics = evaluate_detection(eval_model, val_loader, device, ann_file=args.val_ann_file,
                                         save_dir=exp_folder if is_last else None)
            map50    = metrics["mAP50"]
            map50_95 = metrics["mAP50_95"]
            loss_parts = (f"ce={extra_avgs['loss_ce']:.4f} bbox={extra_avgs['loss_bbox']:.4f} "
                          f"giou={extra_avgs['loss_giou']:.4f}  " if extra_avgs else "")
            print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  {loss_parts}"
                  f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}  lr={lr_now:.6f}")
            if extra_avgs:
                row = [epoch, avg_loss, extra_avgs["loss_ce"], extra_avgs["loss_bbox"],
                       extra_avgs["loss_giou"], map50, map50_95, lr_now]
            else:
                row = [epoch, avg_loss, map50, map50_95, lr_now]
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if tb_writer is not None:
                tb_writer.add_scalar("Val/mAP50", map50, epoch)
                tb_writer.add_scalar("Val/mAP50_95", map50_95, epoch)
            return map50
        else:
            eval_interval = max(1, int(args.eval_interval))
            print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  eval=skipped (every {eval_interval} epochs)  lr={lr_now:.6f}")
            if extra_avgs:
                row = [epoch, avg_loss, extra_avgs["loss_ce"], extra_avgs["loss_bbox"],
                       extra_avgs["loss_giou"], "", "", lr_now]
            else:
                row = [epoch, avg_loss, "", "", lr_now]
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return None
    else:  # segment
        if do_eval:
            print(f"[Eval ][epoch {epoch+1}/{args.epochs}] running segmentation evaluation...")
            is_last = (epoch + 1) == args.epochs
            metrics = evaluate_segmentation(eval_model, val_loader, device, ann_file=args.val_ann_file,
                                            save_dir=exp_folder if is_last else None)
            print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  "
                  f"box_mAP50={metrics['box_mAP50']:.4f}  mask_mAP50={metrics['mask_mAP50']:.4f}  lr={lr_now:.6f}")
            if extra_avgs:
                row = [epoch, avg_loss, extra_avgs["loss_ce"], extra_avgs["loss_bbox"],
                       extra_avgs["loss_giou"], extra_avgs.get("loss_mask", ""),
                       extra_avgs.get("loss_dice", ""),
                       metrics["box_mAP50"], metrics["box_mAP50_95"],
                       metrics["mask_mAP50"], metrics["mask_mAP50_95"], lr_now]
            else:
                row = [epoch, avg_loss,
                       metrics["box_mAP50"], metrics["box_mAP50_95"],
                       metrics["mask_mAP50"], metrics["mask_mAP50_95"], lr_now]
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if tb_writer is not None:
                tb_writer.add_scalar("Val/box_mAP50", metrics["box_mAP50"], epoch)
                tb_writer.add_scalar("Val/box_mAP50_95", metrics["box_mAP50_95"], epoch)
                tb_writer.add_scalar("Val/mask_mAP50", metrics["mask_mAP50"], epoch)
                tb_writer.add_scalar("Val/mask_mAP50_95", metrics["mask_mAP50_95"], epoch)
            return metrics["mask_mAP50"] if not extra_avgs else metrics["box_mAP50"] + metrics["mask_mAP50"]
        else:
            eval_interval = max(1, int(args.eval_interval))
            print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  eval=skipped (every {eval_interval} epochs)  lr={lr_now:.6f}")
            if extra_avgs:
                row = [epoch, avg_loss, extra_avgs["loss_ce"], extra_avgs["loss_bbox"],
                       extra_avgs["loss_giou"], extra_avgs.get("loss_mask", ""),
                       extra_avgs.get("loss_dice", ""),
                       "", "", "", "", lr_now]
            else:
                row = [epoch, avg_loss, "", "", "", "", lr_now]
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return None


def _save_checkpoint_and_check(model, optimizer, scheduler, ema, args,
                               epoch, best_metric, best_epoch, primary,
                               early_stop, last_ckpt_path, best_ckpt_path):
    """Save checkpoint. Returns (best_metric, best_epoch, should_stop)."""
    if (primary is not None) and (primary > best_metric):
        best_metric = primary
        best_epoch = epoch

    ckpt = {
        "epoch": epoch, "model_state": model.state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "best_metric": best_metric, "best_epoch": best_epoch,
        "args": vars(args),
    }
    torch.save(ckpt, last_ckpt_path)
    if epoch == best_epoch:
        torch.save(ckpt, best_ckpt_path)

    should_stop = (primary is not None and early_stop.step(primary, epoch))
    return best_metric, best_epoch, should_stop


# ======================================================================
# Detection / segmentation training loops
# ======================================================================

def _train_detect_segment(args, device, exp_folder, weights_folder, task: str):
    (build_coco_dataloaders, evaluate_detection, evaluate_segmentation,
     build_detection_model, build_segmentation_model) = _import_det_seg()

    load_masks = (task == "segment")
    train_loader, val_loader, num_classes = build_coco_dataloaders(
        train_img_dir=args.train_img_dir,
        train_ann_file=args.train_ann_file,
        val_img_dir=args.val_img_dir,
        val_ann_file=args.val_ann_file,
        batch_size=args.batch_size,
        load_masks=load_masks,
        num_workers=args.num_workers,
    )
    print(f"[INFO] COCO dataset: {num_classes} foreground classes.")

    if task == "detect":
        model = build_detection_model(
            backbone_name=args.model,
            num_classes=num_classes,
            backbone_weights=args.weights,
            freeze_backbone=(args.freeze_layers != "none"),
        )
    else:
        model = build_segmentation_model(
            backbone_name=args.model,
            num_classes=num_classes,
            backbone_weights=args.weights,
            freeze_backbone=(args.freeze_layers != "none"),
        )
    model.to(device)

    pg = make_param_groups(model, args.lr, args.backbone_lr_scale, weight_decay=0.05)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.0)
    lf = make_cosine_lr(args.epochs, args.lrf, args.warmup_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[EMA] enabled, decay={args.ema_decay}")

    # CSV header
    metrics_path = os.path.join(exp_folder, "metrics.csv")
    if task == "detect":
        header = ["epoch", "train_loss", "mAP50", "mAP50_95", "lr"]
    else:
        header = ["epoch", "train_loss", "box_mAP50", "box_mAP50_95", "mask_mAP50", "mask_mAP50_95", "lr"]
    if not (getattr(args, 'resume', '') and os.path.exists(metrics_path)):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    st = _setup_training_state(args, model, optimizer, scheduler, device, ema, weights_folder, exp_folder)

    printer = ConsolePrinter()
    for epoch in range(st["start_epoch"], args.epochs):
        avg_loss, _, _ = _train_one_epoch_det(
            model, train_loader, optimizer, device, epoch, args,
            printer, st["scaler"], st["use_amp"], st["accumulate_steps"], ema,
            loss_key=None, max_norm=1.0,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        do_eval = ((epoch + 1) % st["eval_interval"] == 0) or ((epoch + 1) == args.epochs)
        eval_model = ema.ema if ema is not None else model

        primary = _eval_and_log_det(
            task, do_eval, eval_model, val_loader, device, args,
            epoch, avg_loss, lr_now, metrics_path, exp_folder,
            evaluate_detection, evaluate_segmentation,
            tb_writer=st["tb_writer"],
        )

        st["best_metric"], st["best_epoch"], should_stop = _save_checkpoint_and_check(
            model, optimizer, scheduler, ema, args,
            epoch, st["best_metric"], st["best_epoch"], primary,
            st["early_stop"], st["last_ckpt_path"], st["best_ckpt_path"],
        )
        if should_stop:
            break

    # Final diagnostic plots: if early stopping triggered before the last configured
    # epoch, the loop never ran evaluation with save_dir. Run it now.
    last_eval_was_final = (epoch + 1 == args.epochs)
    if not last_eval_was_final and os.path.exists(st["best_ckpt_path"]):
        print(f"[INFO] Generating final diagnostic plots (early stop at epoch {epoch+1})...")
        best_ckpt = torch.load(st["best_ckpt_path"], map_location=device)
        eval_model_final = ema.ema if ema is not None else model
        if ema is not None and best_ckpt.get("ema_state") is not None:
            ema.load_state_dict(best_ckpt["ema_state"])
        else:
            model.load_state_dict(best_ckpt["model_state"], strict=True)
        eval_model_final.eval()
        if task == "detect":
            evaluate_detection(eval_model_final, val_loader, device,
                               ann_file=args.val_ann_file, save_dir=exp_folder)
        else:
            evaluate_segmentation(eval_model_final, val_loader, device,
                                  ann_file=args.val_ann_file, save_dir=exp_folder)

    if st["tb_writer"] is not None:
        st["tb_writer"].close()
    print(f"Training done. Best metric={st['best_metric']:.4f} at epoch={st['best_epoch']}")
    print(f"Last: {st['last_ckpt_path']}  Best: {st['best_ckpt_path']}")
    if args.lr > 1e-3:
        print(f"[WARN] --lr={args.lr} may be too large for AdamW detect/segment. "
             f"Recommended: 1e-4 ~ 5e-4.")


def _train_detr(args, device, exp_folder, weights_folder, task: str):
    """Training loop for DETR-style detection / segmentation."""
    (build_coco_dataloaders, evaluate_detection, evaluate_segmentation,
     build_detr_model) = _import_detr()

    load_masks = (task == "segment")
    train_loader, val_loader, num_classes = build_coco_dataloaders(
        train_img_dir=args.train_img_dir,
        train_ann_file=args.train_ann_file,
        val_img_dir=args.val_img_dir,
        val_ann_file=args.val_ann_file,
        batch_size=args.batch_size,
        load_masks=load_masks,
    )
    print(f"[INFO] COCO dataset: {num_classes} foreground classes (DETR {task}).")

    model = build_detr_model(
        backbone_name=args.model,
        num_classes=num_classes,
        task=task,
        backbone_weights=args.weights,
        freeze_backbone=args.freeze_layers,
        d_model=args.d_model,
        nhead=8,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        num_queries=args.num_queries,
        num_dn_groups=args.num_dn_groups,
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    model.to(device)

    pg = make_param_groups(model, args.lr, args.backbone_lr_scale, weight_decay=0.05)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.0)
    lf = make_cosine_lr(args.epochs, args.lrf, args.warmup_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[EMA] enabled, decay={args.ema_decay}")

    # CSV header
    if task == "detect":
        header = ["epoch", "loss_total", "loss_ce", "loss_bbox", "loss_giou", "mAP50", "mAP50_95", "lr"]
    else:
        header = ["epoch", "loss_total", "loss_ce", "loss_bbox", "loss_giou",
                  "loss_mask", "loss_dice",
                  "box_mAP50", "box_mAP50_95", "mask_mAP50", "mask_mAP50_95", "lr"]
    metrics_path = os.path.join(exp_folder, "metrics.csv")
    if not (getattr(args, 'resume', '') and os.path.exists(metrics_path)):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    detr_extra_keys = ("loss_ce", "loss_bbox", "loss_giou", "loss_mask", "loss_dice")
    st = _setup_training_state(args, model, optimizer, scheduler, device, ema, weights_folder, exp_folder)

    printer = ConsolePrinter()
    for epoch in range(st["start_epoch"], args.epochs):
        avg_loss, extra_avgs, _ = _train_one_epoch_det(
            model, train_loader, optimizer, device, epoch, args,
            printer, st["scaler"], st["use_amp"], st["accumulate_steps"], ema,
            loss_key="loss_total", max_norm=0.1, extra_loss_keys=detr_extra_keys,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        do_eval = ((epoch + 1) % st["eval_interval"] == 0) or ((epoch + 1) == args.epochs)
        eval_model = ema.ema if ema is not None else model

        primary = _eval_and_log_det(
            task, do_eval, eval_model, val_loader, device, args,
            epoch, avg_loss, lr_now, metrics_path, exp_folder,
            evaluate_detection, evaluate_segmentation,
            extra_avgs=extra_avgs, tb_writer=st["tb_writer"],
        )

        st["best_metric"], st["best_epoch"], should_stop = _save_checkpoint_and_check(
            model, optimizer, scheduler, ema, args,
            epoch, st["best_metric"], st["best_epoch"], primary,
            st["early_stop"], st["last_ckpt_path"], st["best_ckpt_path"],
        )
        if should_stop:
            break

    # Final diagnostic plots: if early stopping triggered before the last configured
    # epoch, the loop never ran evaluation with save_dir. Run it now.
    last_eval_was_final = (epoch + 1 == args.epochs)
    if not last_eval_was_final and os.path.exists(st["best_ckpt_path"]):
        print(f"[INFO] Generating final diagnostic plots (early stop at epoch {epoch+1})...")
        best_ckpt = torch.load(st["best_ckpt_path"], map_location=device)
        eval_model_final = ema.ema if ema is not None else model
        if ema is not None and best_ckpt.get("ema_state") is not None:
            ema.load_state_dict(best_ckpt["ema_state"])
        else:
            model.load_state_dict(best_ckpt["model_state"], strict=True)
        eval_model_final.eval()
        if task == "detect":
            evaluate_detection(eval_model_final, val_loader, device,
                               ann_file=args.val_ann_file, save_dir=exp_folder)
        else:
            evaluate_segmentation(eval_model_final, val_loader, device,
                                  ann_file=args.val_ann_file, save_dir=exp_folder)

    if st["tb_writer"] is not None:
        st["tb_writer"].close()
    print(f"Training done. Best metric={st['best_metric']:.4f} at epoch={st['best_epoch']}")
    print(f"Last: {st['last_ckpt_path']}  Best: {st['best_ckpt_path']}")
    if args.lr > 1e-3:
        print(f"[WARN] --lr={args.lr} may be too large for AdamW DETR. "
              f"Recommended: 1e-4 ~ 5e-4.")

    from tools.plot_metrics import plot_detr_metrics
    plot_detr_metrics(metrics_path, exp_folder)
    print(f"[INFO] DETR plots saved to {exp_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- task ----
    parser.add_argument('--task', type=str, default='detect',
                        choices=['classify', 'detect', 'segment', 'detr_detect', 'detr_segment'],
                        help='Task type: classify | detect | segment | detr_detect | detr_segment')

    # ---- common ----
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=0.001,
                        help='Learning rate. Recommended: 0.001 for detection/segmentation with AdamW; 0.01 for classification with SGD.')
    parser.add_argument('--lrf',          type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=int,   default=5,
                        help='Linear LR warmup epochs before cosine decay. '
                             'LR rises from 0 to --lr over this many epochs. '
                             'Recommended: 5 for classify, 5 for detect/segment, 10 for detr. '
                             'Set 0 to disable.')
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1,
                        help='LR multiplier for backbone parameters relative to head. '
                             'backbone_lr = --lr * --backbone-lr-scale. '
                             'Applies only when backbone is not fully frozen. '
                             'Default 0.1 (backbone trains at 1/10 of head LR).')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay for model weight averaging. '
                             'shadow_w = decay * shadow_w + (1-decay) * model_w. '
                             'EMA weights are used for evaluation and saved in checkpoints. '
                             'Set 0 to disable EMA. '
                             'Recommended: 0.9999 for large-batch runs, 0.999 for small-batch / short runs.')
    parser.add_argument('--model',      type=str,   default="swin_base_patch4_window7_224",
                        help='Backbone name. ViT: vit_base_patch16_224_in21k | vit_large_patch16_224_in21k | vit_huge_patch14_224_in21k. '
                             'Swin: swin_tiny_patch4_window7_224 | swin_small_patch4_window7_224 | swin_base_patch4_window7_224')
    parser.add_argument('--weights',    type=str,   default='weights/swin_base_patch4_window7_224_22k.pth',
                        help='Pretrained backbone weights path; pass empty string to skip')
    def _parse_freeze(x):
        v = x.lower()
        if v in ("true", "all"):    return "all"
        if v in ("false", "none"):  return "none"
        if v == "partial":          return "partial"
        raise argparse.ArgumentTypeError(
            f"--freeze-layers: expected all/partial/none (or True/False), got '{x}'")
    parser.add_argument('--freeze-layers', type=_parse_freeze, default="none",
                        help='Freeze backbone layers. '
                             'all: freeze everything (default); '
                             'partial: freeze stage0+1, unfreeze stage2+3 (DETR only); '
                             'none: train full backbone. '
                             'Example: --freeze-layers partial')
    parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='[detect/segment] Run evaluation every N epochs (last epoch is always evaluated)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience: stop if metric does not improve for N epochs. '
                             'Set 0 to disable. Default: 10.')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable AMP (Automatic Mixed Precision) training. '
                             'Default: on. Use --no-amp to disable.')
    parser.add_argument('--accumulate-steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch size = batch-size * accumulate-steps. '
                             'Default: 1 (no accumulation).')
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable TensorBoard logging to exp_folder/tb_logs/. '
                             'Default: on. Use --no-tensorboard to disable.')

    parser.add_argument('--num-workers', type=int, default=-1,
                        help='DataLoader worker processes. -1 = auto (min(cpu_count, batch_size, 8)).')

    # ---- resume (detect / segment / detr_detect / detr_segment only) ----
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a checkpoint (.pth/.pt) to resume training from. '
                             'Example: run/train/exp33/weights/last.pth  '
                             'The exp folder is inferred automatically from the path. '
                             '--epochs is the *total* target (not additional epochs). '
                             'Not supported for --task classify.')

    # ---- classify only ----
    parser.add_argument('--data-path',  type=str,   default="Plant_Leaf_Disease",
                        help='[classify] Dataset root directory (ImageFolder style)')

    # ---- detect / segment only ----
    parser.add_argument('--train-img-dir',  type=str, default="data/uadetrac-1/train",
                        help='[detect/segment] Training images directory')
    parser.add_argument('--train-ann-file', type=str, default="data/uadetrac-1/train/_annotations.coco.json",
                        help='[detect/segment] Training COCO annotation JSON')
    parser.add_argument('--val-img-dir',    type=str, default="data/uadetrac-1/valid",
                        help='[detect/segment] Validation images directory')
    parser.add_argument('--val-ann-file',   type=str, default="data/uadetrac-1/valid/_annotations.coco.json",
                        help='[detect/segment] Validation COCO annotation JSON')

    # ---- DETR only (ignored by other tasks) ----
    parser.add_argument('--num-queries',    type=int,   default=100,
                        help='[detr] Number of object queries')
    parser.add_argument('--d-model',        type=int,   default=256,
                        help='[detr] Transformer hidden dimension')
    parser.add_argument('--num-enc-layers', type=int,   default=4,
                        help='[detr] Encoder layers')
    parser.add_argument('--num-dec-layers', type=int,   default=4,
                        help='[detr] Decoder layers')
    parser.add_argument('--num-dn-groups',  type=int,   default=2,
                        help='[detr] Number of DN denoising groups')
    parser.add_argument('--cost-class',     type=float, default=1.0,
                        help='[detr] Matching cost weight: classification')
    parser.add_argument('--cost-bbox',      type=float, default=5.0,
                        help='[detr] Matching cost weight: L1 bbox')
    parser.add_argument('--cost-giou',      type=float, default=2.0,
                        help='[detr] Matching cost weight: GIoU')
    parser.add_argument('--min-size',       type=int,   default=800,
                        help='[detr] Minimum image size for input transform (default: 800)')
    parser.add_argument('--max-size',       type=int,   default=1333,
                        help='[detr] Maximum image size for input transform (default: 1333)')

    opt = parser.parse_args()
    main(opt)
  


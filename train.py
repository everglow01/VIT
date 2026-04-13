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
                         get_model_factory, extract_state_dict, make_cosine_lr)
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
        batch_size=args.batch_size
    )

    model = build_model_and_prepare(args, device, num_classes)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = make_cosine_lr(args.epochs, args.lrf, args.warmup_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    last_ckpt_path = os.path.join(weights_folder, "last.pth")
    best_ckpt_path = os.path.join(weights_folder, "best.pth")
    best_val_acc = -1.0
    best_epoch = -1

    printer = ConsolePrinter()
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        print()
        print(printer.train_header(colored=True))
        train_loss, train_acc = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader,
            device=device, epoch=epoch, epochs=args.epochs,
            loss_function=loss_function, printer=printer
        )
        scheduler.step()

        print(printer.val_header(colored=True))
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(
            model=model, data_loader=val_loader, device=device,
            epoch=epoch, epochs=args.epochs, num_classes=num_classes,
            loss_function=loss_function, printer=printer
        )

        lr_now = optimizer.param_groups[0]["lr"]
        val_acc_value  = float(val_acc.item())  if hasattr(val_acc,  "item") else float(val_acc)
        train_acc_value = float(train_acc.item()) if hasattr(train_acc, "item") else float(train_acc)

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc_value, val_loss, val_acc_value, val_p, val_r, val_f1, lr_now])

        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc, "args": vars(args),
        }
        torch.save(ckpt, last_ckpt_path)

        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            best_epoch = epoch
            torch.save(ckpt, best_ckpt_path)

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


def _load_resume_ckpt(resume_path, model, optimizer, scheduler, device, args):
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

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    lf = make_cosine_lr(args.epochs, args.lrf, args.warmup_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # CSV header
    metrics_path = os.path.join(exp_folder, "metrics.csv")
    if task == "detect":
        header = ["epoch", "train_loss", "mAP50", "mAP50_95", "lr"]
    else:
        header = ["epoch", "train_loss", "box_mAP50", "box_mAP50_95", "mask_mAP50", "mask_mAP50_95", "lr"]
    # 续训时 CSV 已存在，追加数据行，不重复写表头；新训始终重建文件
    if not (getattr(args, 'resume', '') and os.path.exists(metrics_path)):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    last_ckpt_path = os.path.join(weights_folder, "last.pth")
    best_ckpt_path = os.path.join(weights_folder, "best.pth")
    best_metric = -1.0
    best_epoch = -1
    eval_interval = max(1, int(args.eval_interval))

    # 续训：恢复模型/优化器/调度器状态，LR 自动从断点延续
    start_epoch = 0
    if getattr(args, 'resume', ''):
        start_epoch, best_metric, best_epoch = _load_resume_ckpt(
            args.resume, model, optimizer, scheduler, device, args)

    printer = ConsolePrinter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
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
        for images, targets in pbar:
            img_size = int(images[0].shape[-1]) if images else 0
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                print(f"\nWARNING: non-finite loss detected at epoch {epoch+1}, skipping this batch: {loss}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pg, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            avg_loss = total_loss / n_batches
            desc = printer.train_desc(epoch + 1, args.epochs, avg_loss, 0.0, img_size)
            pbar.set_description_str(desc)

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        do_eval = ((epoch + 1) % eval_interval == 0) or ((epoch + 1) == args.epochs)

        if task == "detect":
            if do_eval:
                print(f"[Eval ][epoch {epoch+1}/{args.epochs}] running detection evaluation...")
                is_last = (epoch + 1) == args.epochs
                metrics = evaluate_detection(model, val_loader, device, ann_file=args.val_ann_file,
                                             save_dir=exp_folder if is_last else None)
                map50    = metrics["mAP50"]
                map50_95 = metrics["mAP50_95"]
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, avg_loss, map50, map50_95, lr_now])
                primary = map50
            else:
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  eval=skipped (every {eval_interval} epochs)  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, avg_loss, "", "", lr_now])
                primary = None
        else:
            if do_eval:
                print(f"[Eval ][epoch {epoch+1}/{args.epochs}] running segmentation evaluation...")
                is_last = (epoch + 1) == args.epochs
                metrics = evaluate_segmentation(model, val_loader, device, ann_file=args.val_ann_file,
                                                save_dir=exp_folder if is_last else None)
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  "
                      f"box_mAP50={metrics['box_mAP50']:.4f}  mask_mAP50={metrics['mask_mAP50']:.4f}  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, avg_loss,
                                            metrics["box_mAP50"], metrics["box_mAP50_95"],
                                            metrics["mask_mAP50"], metrics["mask_mAP50_95"], lr_now])
                primary = metrics["mask_mAP50"]
            else:
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}  eval=skipped (every {eval_interval} epochs)  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, avg_loss, "", "", "", "", lr_now])
                primary = None

        if (primary is not None) and (primary > best_metric):
            best_metric = primary
            best_epoch = epoch

        # best_metric/best_epoch 写入 checkpoint，供续训时恢复
        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_metric": best_metric, "best_epoch": best_epoch,
            "args": vars(args),
        }
        torch.save(ckpt, last_ckpt_path)
        if epoch == best_epoch:
            torch.save(ckpt, best_ckpt_path)

    print(f"Training done. Best metric={best_metric:.4f} at epoch={best_epoch}")
    print(f"Last: {last_ckpt_path}  Best: {best_ckpt_path}")
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
    )
    model.to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    lf = make_cosine_lr(args.epochs, args.lrf, args.warmup_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # CSV header
    if task == "detect":
        header = ["epoch", "loss_total", "loss_ce", "loss_bbox", "loss_giou", "mAP50", "mAP50_95", "lr"]
    else:
        header = ["epoch", "loss_total", "loss_ce", "loss_bbox", "loss_giou",
                  "loss_mask", "loss_dice",
                  "box_mAP50", "box_mAP50_95", "mask_mAP50", "mask_mAP50_95", "lr"]
    metrics_path = os.path.join(exp_folder, "metrics.csv")
    # 续训时 CSV 已存在，追加数据行，不重复写表头；新训始终重建文件
    if not (getattr(args, 'resume', '') and os.path.exists(metrics_path)):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    last_ckpt_path = os.path.join(weights_folder, "last.pth")
    best_ckpt_path = os.path.join(weights_folder, "best.pth")
    best_metric = -1.0
    best_epoch = -1
    eval_interval = max(1, int(args.eval_interval))

    # 续训：恢复模型/优化器/调度器状态，LR 自动从断点延续
    start_epoch = 0
    if getattr(args, 'resume', ''):
        start_epoch, best_metric, best_epoch = _load_resume_ckpt(
            args.resume, model, optimizer, scheduler, device, args)

    printer = ConsolePrinter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_bbox = 0.0
        total_giou = 0.0
        total_mask = 0.0
        total_dice = 0.0
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
        for images, targets in pbar:
            img_size = int(images[0].shape[-1]) if images else 0
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = loss_dict["loss_total"]

            if not torch.isfinite(loss):
                print(f"\nWARNING: non-finite loss at epoch {epoch+1}, skipping batch: {loss}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pg, max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += loss_dict["loss_ce"].item()
            total_bbox += loss_dict["loss_bbox"].item()
            total_giou += loss_dict["loss_giou"].item()
            if "loss_mask" in loss_dict:
                total_mask += loss_dict["loss_mask"].item()
                total_dice += loss_dict["loss_dice"].item()
            n_batches += 1

            avg_loss = total_loss / n_batches
            desc = printer.train_desc(epoch + 1, args.epochs, avg_loss, 0.0, img_size)
            pbar.set_description_str(desc)

        scheduler.step()
        avg = lambda x: x / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        do_eval = ((epoch + 1) % eval_interval == 0) or ((epoch + 1) == args.epochs)

        if task == "detect":
            if do_eval:
                print(f"[Eval ][epoch {epoch+1}/{args.epochs}] running DETR detection evaluation...")
                is_last = (epoch + 1) == args.epochs
                metrics = evaluate_detection(model, val_loader, device, ann_file=args.val_ann_file,
                                             save_dir=exp_folder if is_last else None)
                map50    = metrics["mAP50"]
                map50_95 = metrics["mAP50_95"]
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg(total_loss):.4f}  "
                      f"ce={avg(total_ce):.4f} bbox={avg(total_bbox):.4f} giou={avg(total_giou):.4f}  "
                      f"mAP50={map50:.4f}  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, avg(total_loss), avg(total_ce), avg(total_bbox), avg(total_giou),
                        map50, map50_95, lr_now,
                    ])
                primary = map50
            else:
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg(total_loss):.4f}  eval=skipped  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, avg(total_loss), avg(total_ce), avg(total_bbox), avg(total_giou),
                        "", "", lr_now,
                    ])
                primary = None
        else:  # segment
            if do_eval:
                print(f"[Eval ][epoch {epoch+1}/{args.epochs}] running DETR segmentation evaluation...")
                is_last = (epoch + 1) == args.epochs
                metrics = evaluate_segmentation(model, val_loader, device, ann_file=args.val_ann_file,
                                                save_dir=exp_folder if is_last else None)
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg(total_loss):.4f}  "
                      f"box_mAP50={metrics['box_mAP50']:.4f}  box_mAP50_95={metrics['box_mAP50_95']:.4f}  "
                      f"mask_mAP50={metrics['mask_mAP50']:.4f}  mask_mAP50_95={metrics['mask_mAP50_95']:.4f}  "
                      f"lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, avg(total_loss), avg(total_ce), avg(total_bbox), avg(total_giou),
                        avg(total_mask), avg(total_dice),
                        metrics["box_mAP50"], metrics["box_mAP50_95"],
                        metrics["mask_mAP50"], metrics["mask_mAP50_95"], lr_now,
                    ])
                # Use box_mAP50 + mask_mAP50 so best.pth tracks progress even when mask AP = 0
                primary = metrics["box_mAP50"] + metrics["mask_mAP50"]
            else:
                print(f"[epoch {epoch+1}/{args.epochs}] loss={avg(total_loss):.4f}  eval=skipped  lr={lr_now:.6f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, avg(total_loss), avg(total_ce), avg(total_bbox), avg(total_giou),
                        avg(total_mask), avg(total_dice),
                        "", "", "", "", lr_now,
                    ])
                primary = None

        if (primary is not None) and (primary > best_metric):
            best_metric = primary
            best_epoch = epoch

        # best_metric/best_epoch 写入 checkpoint，供续训时恢复
        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_metric": best_metric, "best_epoch": best_epoch,
            "args": vars(args),
        }
        torch.save(ckpt, last_ckpt_path)
        if epoch == best_epoch:
            torch.save(ckpt, best_ckpt_path)

    print(f"Training done. Best metric={best_metric:.4f} at epoch={best_epoch}")
    print(f"Last: {last_ckpt_path}  Best: {best_ckpt_path}")
    if args.lr > 1e-3:
        print(f"[WARN] --lr={args.lr} may be too large for AdamW DETR. "
              f"Recommended: 1e-4 ~ 5e-4.")

    from tools.plot_metrics import plot_detr_metrics
    plot_detr_metrics(metrics_path, exp_folder)
    print(f"[INFO] DETR plots saved to {exp_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- task ----
    parser.add_argument('--task', type=str, default='segment',
                        choices=['classify', 'detect', 'segment', 'detr_detect', 'detr_segment'],
                        help='Task type: classify | detect | segment | detr_detect | detr_segment')

    # ---- common ----
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch-size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=0.001,
                        help='Learning rate. Recommended: 0.001 for detection/segmentation with AdamW; 0.01 for classification with SGD.')
    parser.add_argument('--lrf',          type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=int,   default=5,
                        help='Linear LR warmup epochs before cosine decay. '
                             'LR rises from 0 to --lr over this many epochs. '
                             'Recommended: 5 for classify, 5 for detect/segment, 10 for detr. '
                             'Set 0 to disable.')
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
    parser.add_argument('--train-img-dir',  type=str, default="data/TOMATO.v7i.coco-segmentation/images/train",
                        help='[detect/segment] Training images directory')
    parser.add_argument('--train-ann-file', type=str, default="data/TOMATO.v7i.coco-segmentation/annotations/_annotations_train_fixed.json",
                        help='[detect/segment] Training COCO annotation JSON')
    parser.add_argument('--val-img-dir',    type=str, default="data/TOMATO.v7i.coco-segmentation/images/valid",
                        help='[detect/segment] Validation images directory')
    parser.add_argument('--val-ann-file',   type=str, default="data/TOMATO.v7i.coco-segmentation/annotations/_annotations_valid_fixed.json",
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

    opt = parser.parse_args()
    main(opt)
  


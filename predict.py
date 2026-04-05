# predict.py
import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from tools.create_exp_folder import create_val_exp_folder
import model.vit_model as vit_models

# ===== shared config =====
IMG_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
CONF_THRESH = 0.9
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

COLORS = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210,  49), (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168),  (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236),  (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200),  (255, 55,  199),
]


def _color(cls_id: int) -> Tuple[int, int, int]:
    return COLORS[cls_id % len(COLORS)]


def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[-1] in IMG_EXTS


def collect_images(input_path: str) -> List[str]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input path not found: {input_path}")
    if os.path.isfile(input_path):
        if not is_image_file(input_path):
            raise ValueError(f"not an image: {input_path}")
        return [input_path]
    imgs = []
    for root, _, files in os.walk(input_path):
        for fn in files:
            fp = os.path.join(root, fn)
            if is_image_file(fp):
                imgs.append(fp)
    imgs.sort()
    if not imgs:
        raise ValueError(f"no images found under: {input_path}")
    return imgs


def _try_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ============================================================
# Classification helpers
# ============================================================

def load_class_indices(json_path: str) -> Optional[Dict[int, str]]:
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    out = {}
    for k, v in m.items():
        try:
            out[int(k)] = v
        except Exception:
            pass
    return out if out else None


def load_coco_label_map(ann_file: str) -> Optional[Dict[int, str]]:
    """
    Build contiguous 1-based label map from COCO categories.
    Training dataset remaps sorted category ids -> labels [1..N].
    """
    if not ann_file or (not os.path.exists(ann_file)):
        return None
    with open(ann_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    categories = ann.get("categories", [])
    if not categories:
        return None

    categories = sorted(categories, key=lambda c: int(c.get("id", 0)))
    out: Dict[int, str] = {}
    for i, cat in enumerate(categories, start=1):
        out[i] = str(cat.get("name", i))
    return out


def normalize_detseg_class_map(class_map: Optional[Dict[int, str]], num_classes: int) -> Optional[Dict[int, str]]:
    """
    Ensure detect/segment class map uses 1-based contiguous labels [1..num_classes].
    If user provides 0-based map [0..num_classes-1], shift it to 1-based.
    """
    if not class_map:
        return None

    expected_1_based = set(range(1, num_classes + 1))
    keys = set(class_map.keys())
    if keys == expected_1_based:
        return class_map

    expected_0_based = set(range(num_classes))
    if keys == expected_0_based:
        return {k + 1: v for k, v in class_map.items()}

    return None


def resolve_detseg_class_map(args, ckpt: Dict, num_classes: int) -> Optional[Dict[int, str]]:
    """
    Priority:
    1) --class-indices (if valid for detect/segment labels)
    2) COCO categories from --ann-file
    3) COCO categories from checkpoint args (val/train ann file)
    """
    user_map = normalize_detseg_class_map(load_class_indices(args.class_indices), num_classes)
    if user_map is not None:
        return user_map

    ann_file = args.ann_file
    if (not ann_file) and isinstance(ckpt, dict):
        old_args = ckpt.get("args", {})
        if isinstance(old_args, dict):
            ann_file = old_args.get("val_ann_file") or old_args.get("train_ann_file") or ""

    coco_map = normalize_detseg_class_map(load_coco_label_map(ann_file), num_classes)
    return coco_map


def load_classify_checkpoint(weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def infer_num_classes(state_dict: Dict) -> Optional[int]:
    w = state_dict.get("head.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return None


def build_val_transform():
    resize_size = int(IMG_SIZE / 224 * 256)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def draw_text_on_image(img: Image.Image, text: str) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _try_font(24)
    pad = 4
    x0, y0 = 6, 6
    bbox = draw.textbbox((x0, y0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad], fill=(0, 0, 0))
    draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
    return img


@torch.no_grad()
def predict_classify(model, img_pil: Image.Image, tfm, device: torch.device):
    model.eval()
    x = tfm(img_pil.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    pred_idx = int(prob.argmax(dim=1).item())
    pred_prob = float(prob[0, pred_idx].item())
    return pred_idx, pred_prob


def run_classify(args, device, exp_folder):
    state_dict = load_classify_checkpoint(args.weights, device)
    num_classes = infer_num_classes(state_dict)
    class_map = load_class_indices(args.class_indices)

    if num_classes is None:
        if class_map is not None:
            num_classes = len(class_map)
        elif args.num_classes is not None:
            num_classes = int(args.num_classes)
        else:
            raise RuntimeError("Cannot infer num_classes. Provide --class-indices or --num-classes.")

    factory = getattr(vit_models, args.model_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Unknown --model-name: {args.model_name}")
    model = factory(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict, strict=False)

    tfm = build_val_transform()
    use_class_name = (class_map is not None) and (len(class_map) == num_classes)
    img_paths = collect_images(args.data)
    print(f"[INFO] Found {len(img_paths)} images.")

    txt_path = os.path.join(exp_folder, "predictions.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("image_path\tpred_id\tpred_name\tprob\n")
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            pred_id, pred_prob = predict_classify(model, img, tfm, device)
            pred_name = class_map.get(pred_id, str(pred_id)) if use_class_name else str(pred_id)
            f.write(f"{p}\t{pred_id}\t{pred_name}\t{pred_prob:.6f}\n")
            if args.draw:
                out_img = draw_text_on_image(img.copy(), f"{pred_name} ({pred_prob:.3f})")
                out_img.save(os.path.join(exp_folder, os.path.basename(p)))

    print(f"[INFO] Saved predictions to: {txt_path}")


# ============================================================
# Detection helpers
# ============================================================

def draw_boxes(img: Image.Image, boxes, labels, scores, class_map=None) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _try_font(16)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        cls_id = int(label)
        color = _color(cls_id)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = class_map.get(cls_id, str(cls_id)) if class_map else str(cls_id)
        text = f"{name} {score:.2f}"
        tb = draw.textbbox((x1, y1 - 18), text, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((x1, y1 - 18), text, fill=(255, 255, 255), font=font)
    return img


@torch.no_grad()
def run_detect(args, device, exp_folder):
    from model.detection_head import build_detection_model

    num_classes = args.num_classes
    if num_classes is None:
        raise ValueError("--num-classes is required for --task detect")

    model = build_detection_model(
        backbone_name=args.model_name,
        num_classes=num_classes,
        backbone_weights="",
        freeze_backbone=False,
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    class_map = resolve_detseg_class_map(args, ckpt, num_classes)
    img_paths = collect_images(args.data)
    print(f"[INFO] Found {len(img_paths)} images.")

    img_out_dir = os.path.join(exp_folder, "images")
    os.makedirs(img_out_dir, exist_ok=True)

    txt_path = os.path.join(exp_folder, "predictions.txt")
    to_tensor = transforms.ToTensor()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("image_path\tlabel\tscore\tx1\ty1\tx2\ty2\n")
        for p in img_paths:
            img_pil = Image.open(p).convert("RGB")
            img_tensor = to_tensor(img_pil).to(device)
            outputs = model([img_tensor])
            out = outputs[0]

            boxes  = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            keep = scores >= CONF_THRESH
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                name = class_map.get(int(label), str(label)) if class_map else str(label)
                f.write(f"{p}\t{name}\t{score:.4f}\t{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}\n")

            if args.draw:
                vis = draw_boxes(img_pil.copy(), boxes, labels, scores, class_map)
                vis.save(os.path.join(img_out_dir, os.path.basename(p)))

    print(f"[INFO] Saved predictions to: {txt_path}")
    if args.draw:
        print(f"[INFO] Saved visualizations to: {img_out_dir}")


# ============================================================
# Segmentation helpers
# ============================================================

def draw_masks(img: Image.Image, boxes, labels, scores, masks, class_map=None) -> Image.Image:
    """Overlay semi-transparent masks and bounding boxes."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        cls_id = int(label)
        r, g, b = _color(cls_id)
        # mask: [1, H, W] float -> binary numpy
        m = (mask[0] > 0.5).astype(np.uint8)  # [H, W]
        # draw filled mask pixels
        ys, xs = np.where(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            draw_ov.point((x, y), fill=(r, g, b, 100))

    img = Image.alpha_composite(img, overlay).convert("RGB")
    # draw boxes on top
    img = draw_boxes(img, boxes, labels, scores, class_map)
    return img


@torch.no_grad()
def run_segment(args, device, exp_folder):
    from model.segmentation_head import build_segmentation_model

    num_classes = args.num_classes
    if num_classes is None:
        raise ValueError("--num-classes is required for --task segment")

    model = build_segmentation_model(
        backbone_name=args.model_name,
        num_classes=num_classes,
        backbone_weights="",
        freeze_backbone=False,
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    class_map = resolve_detseg_class_map(args, ckpt, num_classes)
    img_paths = collect_images(args.data)
    print(f"[INFO] Found {len(img_paths)} images.")

    img_out_dir = os.path.join(exp_folder, "images")
    os.makedirs(img_out_dir, exist_ok=True)

    txt_path = os.path.join(exp_folder, "predictions.txt")
    to_tensor = transforms.ToTensor()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("image_path\tlabel\tscore\tx1\ty1\tx2\ty2\n")
        for p in img_paths:
            img_pil = Image.open(p).convert("RGB")
            img_tensor = to_tensor(img_pil).to(device)
            outputs = model([img_tensor])
            out = outputs[0]

            boxes  = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            masks  = out["masks"].cpu().numpy()   # [N, 1, H, W]

            keep = scores >= CONF_THRESH
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            masks = masks[keep]

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                name = class_map.get(int(label), str(label)) if class_map else str(label)
                f.write(f"{p}\t{name}\t{score:.4f}\t{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}\n")

            if args.draw:
                vis = draw_masks(img_pil.copy(), boxes, labels, scores, masks, class_map)
                vis.save(os.path.join(img_out_dir, os.path.basename(p)))

    print(f"[INFO] Saved predictions to: {txt_path}")
    if args.draw:
        print(f"[INFO] Saved visualizations to: {img_out_dir}")


# ============================================================
# Entry point
# ============================================================

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    exp_folder = create_val_exp_folder()
    os.makedirs(exp_folder, exist_ok=True)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    if args.task == "classify":
        run_classify(args, device, exp_folder)
    elif args.task == "detect":
        run_detect(args, device, exp_folder)
    elif args.task == "segment":
        run_segment(args, device, exp_folder)
    else:
        raise ValueError(f"Unknown --task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- task ----
    parser.add_argument("--task", type=str, default="segment",
                        choices=["classify", "detect", "segment"])

    # ---- common ----
    parser.add_argument("--data",       type=str, default="data/TOMATO.v5i.coco-segmentation/valid")
    parser.add_argument("--weights",    type=str, default="run/train/exp16/weights/best.pth")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224_in21k")
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--draw",       action="store_true", default=True)
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Required for detect/segment; optional for classify")
    parser.add_argument("--ann-file", type=str, default="data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json",
                        help="Optional COCO annotation JSON for detect/segment class names")

    # ---- classify only ----
    parser.add_argument("--class-indices", type=str, default="run/train/exp/class_indices.json")

    args = parser.parse_args()
    main(args)

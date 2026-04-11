# Vision Transformer + Swin Transformer for Classify, Detection and Segmentation

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-Classify%20%7C%20Detect%20%7C%20Segment%20%7C%20DETR-2EA44F">
  <img alt="License Notice" src="https://img.shields.io/badge/Usage-Non--Commercial%20Only-8A2BE2">
</p>

<p align="center">
  <b>🌿 ViT/Swin Learning Project for Classification, Detection, and Segmentation</b>
</p>

<p align="center">
  ✨ Clean training workflow • 📊 Built-in metrics • 🧩 COCO support • 🖼️ Visualization • 🔄 Checkpoint Resume • 🎯 DN-DETR
</p>

> <font color="#e67e22"><b>Notice:</b></font> This repository is intended for learning and research demonstrations.

## Important Notice (Non-Commercial Only)

This repository is provided strictly for learning, research, and educational demonstration.

Commercial use is strictly prohibited in any form, including but not limited to:
- Selling this codebase or trained models
- Integrating this project into paid products or services
- Using this project in profit-oriented production environments

By using this repository, you agree to these restrictions.

## 🧭 Overview

This project brings Vision Transformer (ViT) and Swin Transformer into **five** classic computer vision tasks — image classification, R-CNN-style object detection, R-CNN-style instance segmentation, and two DN-DETR-based variants (DETR detection and DETR segmentation) — all under one unified training and inference entry point.

For the R-CNN-style path, the backbone (ViT or Swin) feeds through an FPN neck into a Fast R-CNN detection head, with an optional Mask R-CNN mask branch layered on top for segmentation. For the **DN-DETR path**, a Swin backbone drives a full Transformer encoder-decoder with denoising queries — skipping proposals entirely and predicting boxes end-to-end via Hungarian matching.

Transfer learning is the default workflow everywhere — freeze the backbone, fine-tune the head, and get results fast even on modest hardware. All five tasks share the same `train.py` / `predict.py` entry point, and outputs (metrics CSVs, plots, checkpoints) land consistently in `run/train/expN/`.

Interrupted a run halfway through? No sweat — the `--resume` flag picks up exactly where you left off, restoring model weights, optimizer momentum, and the LR schedule in one shot.

## ✨ Key Features

- **Multiple backbones:** ViT (base / large / huge) and Swin (tiny / small / base)
- **Five tasks in one entry point:** `classify`, `detect`, `segment`, `detr_detect`, `detr_segment`
- **Two detection paradigms:**
  - ViT/Swin + FPN + Fast R-CNN head (classic two-stage style)
  - Swin + DN-DETR (end-to-end Transformer with denoising queries, no proposals needed)
- COCO-format training and evaluation with pycocotools mAP (mAP@0.5, mAP@0.5:0.95)
- Classification metrics: loss, accuracy, macro precision / recall / F1 per epoch
- Rich post-training evaluation charts for detection/segmentation:
  - PR curves, F1-Confidence curve, per-class AP bar chart
  - Confusion matrix (with background class), calibration curve
  - Scale analysis (small / medium / large), mask IoU analysis (segmentation)
- **Checkpoint resume (`--resume`):** resume any detect/segment/detr_detect/detr_segment run from a saved `.pth`, restoring full optimizer and scheduler state — `--epochs` is the absolute target, not "extra epochs"
- Auto-increment experiment folders (`run/train/expN`), with resume reusing the original folder
- **Optional NMS post-processing in inference** (`--nms`, default on): per-class Non-Maximum Suppression across all four non-classify tasks
- Inference with optional visualization (`--draw`) and TSV prediction export

### 🔍 About R-CNN-style Detection / Segmentation
Supports both ViT and Swin backbones. A unified FPN neck aggregates multi-scale features before a torchvision-style Fast R-CNN head for regression and classification. For segmentation, a Mask R-CNN mask branch predicts pixel-level instance masks on top.

### 🎯 About DN-DETR Detection / Segmentation
Uses a **Swin backbone** (tiny/small/base) as a pure feature extractor, feeding three scales (P3/P4/P5) into a full Transformer encoder-decoder. Denoising (DN) queries during training teach the decoder to denoise noisy GT boxes, which dramatically accelerates convergence compared to vanilla DETR. At inference, box predictions arrive in one pass — no anchor boxes, no NMS required in principle (though `--nms` can still help with undertrained models). The `detr_segment` variant adds a lightweight mask branch on top of decoder queries for instance masks.

Backbone freezing for DETR supports three modes:
- `True` / `"all"` — freeze everything (fastest, good for small datasets)
- `"partial"` — freeze stage 0+1, keep stage 2+3 and all norms trainable. When training a DETR-style model, it is more recommended (better accuracy, costs more VRAM)
- `False` / `"none"` — full end-to-end training

## 🗂️ Project Structure

- `train.py`: Unified entry for all five tasks
- `predict.py`: Inference for single image or directory (all five tasks)
- `model/vit_model.py`: ViT backbone implementation
- `model/swin_model.py`: Swin backbone implementation
- `model/detection_head.py`: Fast R-CNN style detection head
- `model/segmentation_head.py`: Mask R-CNN style segmentation head
- `model/detr_head.py`: DN-DETR Transformer encoder-decoder head
- `model/mask_branch.py`: Mask branch for DETR segmentation
- `model/swin_detr.py`: Swin + DN-DETR full model wiring
- `tools/my_dataset.py`: Classification dataset and dataloaders
- `tools/coco_dataset.py`: COCO dataset and dataloaders
- `tools/utils.py`: Training/evaluation helpers (including `apply_nms`)
- `tools/matcher.py`: Hungarian matcher for DETR training
- `tools/detr_loss.py`: SetCriterion loss for DETR (CE + L1 bbox + GIoU + mask/dice)
- `tools/plot_metrics.py`: Metric and confusion-matrix plotting
- `tools/create_exp_folder.py`: Experiment folder management

## ⚙️ Installation

Recommended Python version: 3.10+

```bash
pip install -r requirements.txt
```

Main dependencies:
- torch
- torchvision
- tqdm
- Pillow
- numpy
- pandas
- matplotlib
- opencv 4.9 or above
- onnx

## 🧪 Data Preparation

### Classification

Use ImageFolder-style data (one class per subfolder).

### Detection / Segmentation (R-CNN and DETR)

Use COCO-format data:
- Image directory (for example `data/val2017`)
- Annotation file (for example `instances_val2017.json`)  


## 📚 Additional Quick Reference (Additive)

This section is added as a supplement and does not replace any existing content above.

### Task Quick Matrix

| Task | Required Input | Main Script | Typical Output |
|------|----------------|-------------|----------------|
| classify | ImageFolder-style dataset | `train.py` / `predict.py` | class id + class name + probability |
| detect | COCO images + COCO annotations | `train.py` / `predict.py` | box + label + score |
| segment | COCO images + COCO annotations (with masks) | `train.py` / `predict.py` | mask + box + label + score |
| detr_detect | COCO images + COCO annotations | `train.py` / `predict.py` | box + label + score (end-to-end) |
| detr_segment | COCO images + COCO annotations (with masks) | `train.py` / `predict.py` | mask + box + label + score (end-to-end) |

### Train Args Quick Reference

| Argument | Task | Meaning |
|----------|------|---------|
| `--task` | all | choose from classify / detect / segment / detr_detect / detr_segment |
| `--model` | train | backbone variant (ViT/Swin) |
| `--weights` | train | pretrained backbone weights |
| `--resume` | detect/segment/detr_* | path to a `.pth` checkpoint to continue training; `--epochs` is the absolute target |
| `--batch-size` | train | batch size (reduce if OOM) |
| `--eval-interval` | detect/segment/detr_* | run evaluation every N epochs (always runs at last epoch) |
| `--freeze-layers` | all | `True` / `False` for ViT/Swin R-CNN; `True` / `"partial"` / `False` for DETR |
| `--num-queries` | detr_* | number of object queries (default 100) |
| `--d-model` | detr_* | Transformer hidden dimension (default 256) |
| `--num-encoder-layers` | detr_* | Transformer encoder depth (default 4) |
| `--num-decoder-layers` | detr_* | Transformer decoder depth (default 4) |
| `--num-dn-groups` | detr_* | DN denoising groups (default 2) |
| `--cost-class` | detr_* | Hungarian matching cost weight for class (default 1.0) |
| `--cost-bbox` | detr_* | Hungarian matching cost weight for L1 bbox (default 5.0) |
| `--cost-giou` | detr_* | Hungarian matching cost weight for GIoU (default 2.0) |

### Predict Args Quick Reference

| Argument | Meaning |
|----------|---------|
| `--task` | inference mode: classify / detect / segment / detr_detect / detr_segment |
| `--data` | single image path or image directory |
| `--weights` | checkpoint used for prediction |
| `--num-classes` | required for detect/segment/detr_* |
| `--ann-file` | optional COCO json for class-name mapping in detect/segment/detr_* |
| `--class-indices` | class name json (mainly for classification) |
| `--draw` | save visualized predictions |
| `--nms` | apply per-class NMS after confidence filtering (default: on); use `--no-nms` to disable |
| `--nms-iou` | IoU threshold for NMS (default: 0.5) |

### Confidence Threshold Policy

- All four non-classify tasks share a global confidence threshold `CONF_THRESH = 0.65` in `predict.py`.
- Only predictions with `score >= CONF_THRESH` are written to `predictions.txt` and drawn on output images.
- After confidence filtering, optional per-class NMS (`--nms`, enabled by default) further removes redundant overlapping boxes using `torchvision.ops.batched_nms`.
- If predictions look empty, try **lowering `CONF_THRESH`** in `predict.py`. If predictions are too crowded with overlapping boxes, try **raising `--nms-iou`** (or disabling NMS entirely with `--no-nms` and relying on threshold alone).
- Note: For DETR tasks, there are effectively two thresholds — the internal model threshold in `swin_detr.py` (line 231, currently `0.65`) and `CONF_THRESH` in `predict.py`. The tighter one acts as the real gate.

### Checkpoint Resume

Training crashed at epoch 87 out of 200? Don't start over. Use `--resume`:

```bash
python train.py \
  --task detr_segment \
  --resume run/train/exp33/weights/last.pth \
  --epochs 200 \   # absolute total — NOT "200 more epochs"
  ... (same data/model args as original run)
```

Behaviour:
- **Exp folder is reused automatically** — inferred from the checkpoint path, no new `expN` folder is created
- **Metrics CSV is appended** — existing rows are preserved, new epochs are added; final plots cover the full history
- **Full state is restored** — model weights, optimizer momentum, LR scheduler state, `best_metric`, `best_epoch`
- **LR note:** Because the cosine schedule stretches over the new `--epochs` total, resuming with a higher epoch target causes a small LR bump at the resume point. This is expected and harmless — the model typically recovers within a few epochs.
- Not supported for `--task classify`.

### Label Name Mapping Priority (Detect/Segment)

When generating class names for detect/segment/detr_* prediction, the mapping priority is:
1. `--class-indices` (if format is valid for detect/segment labels)
2. COCO categories from `--ann-file`
3. Annotation path stored in checkpoint args (fallback)

### Prediction File Format Examples

Classification `predictions.txt` (TSV):

```text
image_path\tpred_id\tpred_name\tprob
path/to/img1.jpg\t2\tTomato__Tomato_Healthy\t0.998731
```

Detection/Segmentation `predictions.txt` (TSV):

```text
image_path\tlabel\tscore\tx1\ty1\tx2\ty2
path/to/img1.jpg\tstem\t0.9321\t120.6\t88.4\t232.5\t210.3
```

### FAQ (Short)

1. **Prediction has objects but wrong class names.**
   - Check `--ann-file` points to the correct COCO json and verify `categories` content/order.

2. **Prediction output is empty.**
   - `CONF_THRESH = 0.65` may be too strict. Lower it in `predict.py` for more recall, or check if the DETR model internal threshold (`swin_detr.py:231`) is also filtering aggressively.

3. **Detect/segment training seems stuck.**
   - Evaluation and data loading can take long; monitor epoch/batch logs and GPU utilization.

4. **DETR loss is high early on / not converging.**
   - DETR-style models typically need more epochs than R-CNN. Try `--freeze-layers partial` to unfreeze Swin stage2+3, or increase `--num-queries` for dense scenes.

5. **Resume failed with "start_epoch >= --epochs".**
   - Your `--epochs` value is too small. If you trained for 150 epochs and saved `last.pth` at epoch 149, you need `--epochs 151` or higher to continue.

6. **Git push fails intermittently.**
   - Usually a network/TLS issue; retry after switching network or DNS.



## 🚀 Training

> Tip: If your GPU memory is limited, lower `--batch-size` first and keep `--eval-interval` larger for smoother training.

### 💡 Tips

- **Freeze the backbone first.** Unless you have a large dataset, keep `--freeze-layers True` and only fine-tune the head. It trains faster and avoids overfitting on small data. For DETR, `"partial"` is a good middle ground once you have initial results.

- **VRAM is the main bottleneck.** ViT/Swin backbones are memory-hungry. If you hit OOM, drop `--batch-size` to 2–4 first. Detection/segmentation needs at least 8 GB; classification is more forgiving.

- **Evaluation runs on the last epoch (and every `--eval-interval` epochs).** The default interval is 10 to keep training fast. Set `--eval-interval 1` if you want per-epoch mAP, but expect slower runs.

- **mAP follows the COCO standard.** Both mAP@0.5 and mAP@0.5:0.95 are computed with pycocotools, so numbers are directly comparable to other COCO benchmarks.

- **Class names come from your COCO JSON.** If labels look wrong in outputs or charts, check the `categories` field in your annotation file.

- **For DETR, use AdamW not SGD.** The code auto-selects AdamW for `detr_*` tasks. If you pass `--lr` values designed for SGD (e.g., 0.001–0.01), you'll get a warning — DETR typically needs `--lr 1e-4` or lower.


### 1) Classification

```bash
python train.py \
  --task classify \
  --data-path Plant_Leaf_Disease \
  --model vit_base_patch16_224_in21k \
  --weights weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.001 \
  --lrf 0.01 \
  --freeze-layers True \
  --device cuda:0
```

### 2) Detection

```bash
python train.py \
  --task detect \
  --train-img-dir data/TOMATO.v5i.coco-segmentation/train \
  --train-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_train.coco.json \
  --val-img-dir data/TOMATO.v5i.coco-segmentation/valid \
  --val-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --model swin_small_patch4_window7_224 \
  --weights weights/swin_small_patch4_window7_224.pth \
  --epochs 100 \
  --eval-interval 1 \
  --batch-size 4 \
  --lr 0.001 \
  --device cuda:0
```

### 3) Segmentation

```bash
python train.py \
  --task segment \
  --train-img-dir data/TOMATO.v5i.coco-segmentation/train \
  --train-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_train.coco.json \
  --val-img-dir data/TOMATO.v5i.coco-segmentation/valid \
  --val-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --model swin_small_patch4_window7_224 \
  --weights weights/swin_small_patch4_window7_224.pth \
  --epochs 100 \
  --eval-interval 1 \
  --batch-size 4 \
  --lr 0.001 \
  --device cuda:0
```

### 4) DN-DETR Detection

End-to-end detection with Swin backbone + Transformer encoder-decoder. No proposals, no anchors — just queries.

```bash
python train.py \
  --task detr_detect \
  --train-img-dir data/TOMATO.v5i.coco-segmentation/train \
  --train-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_train.coco.json \
  --val-img-dir data/TOMATO.v5i.coco-segmentation/valid \
  --val-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --model swin_tiny_patch4_window7_224 \
  --weights weights/swin_tiny_patch4_window7_224.pth \
  --epochs 150 \
  --eval-interval 5 \
  --batch-size 4 \
  --lr 1e-4 \
  --freeze-layers True \
  --num-queries 100 \
  --device cuda:0
```

### 5) DN-DETR Segmentation

Same as DETR detection but with an extra mask branch. Predicts boxes **and** per-instance pixel masks end-to-end.

```bash
python train.py \
  --task detr_segment \
  --train-img-dir data/TOMATO.v5i.coco-segmentation/train \
  --train-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_train.coco.json \
  --val-img-dir data/TOMATO.v5i.coco-segmentation/valid \
  --val-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --model swin_tiny_patch4_window7_224 \
  --weights weights/swin_tiny_patch4_window7_224.pth \
  --epochs 200 \
  --eval-interval 5 \
  --batch-size 4 \
  --lr 1e-4 \
  --freeze-layers True \
  --num-queries 100 \
  --device cuda:0
```

### 6) Resume from Checkpoint

Training interrupted? Pick up exactly where you left off:

```bash
python train.py \
  --task detr_segment \
  --resume run/train/exp33/weights/last.pth \
  --epochs 200 \
  --train-img-dir data/TOMATO.v5i.coco-segmentation/train \
  --train-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_train.coco.json \
  --val-img-dir data/TOMATO.v5i.coco-segmentation/valid \
  --val-ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --model swin_tiny_patch4_window7_224 \
  --device cuda:0
```

> Use `last.pth` rather than `best.pth` for resume — `last.pth` always carries the latest optimizer and scheduler state.

### 7) Available Backbone Names

ViT (for classify / detect / segment):
- `vit_base_patch16_224_in21k`
- `vit_base_patch32_224_in21k`
- `vit_large_patch16_224_in21k`
- `vit_large_patch32_224_in21k`
- `vit_huge_patch14_224_in21k`

Swin (for all tasks including detr_*):
- `swin_tiny_patch4_window7_224`
- `swin_small_patch4_window7_224`
- `swin_base_patch4_window7_224`

## 📌 Usage Examples

> Quick start suggestion: run the 1-epoch smoke tests first, then scale epochs and batch size.

### Example A: Quick Classification Smoke Test (1 epoch)

```bash
python train.py \
  --task classify \
  --data-path Plant_Leaf_Disease \
  --epochs 1 \
  --batch-size 16 \
  --device cuda:0
```

### Example B: Quick Detection Smoke Test (1 epoch)

```bash
python train.py \
  --task detect \
  --train-img-dir data/val2017 \
  --train-ann-file data/annotations_trainval2017/annotations/instances_val2017.json \
  --val-img-dir data/val2017 \
  --val-ann-file data/annotations_trainval2017/annotations/instances_val2017.json \
  --epochs 1 \
  --eval-interval 1 \
  --batch-size 2 \
  --device cuda:0
```

### Example C: Predict a Single Image

```bash
python predict.py \
  --task segment \
  --data path/to/target.jpg \
  --weights run/train/exp22/weights/best.pth \
  --model-name swin_small_patch4_window7_224 \
  --num-classes 2 \
  --ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --device cuda:0 \
  --draw
```

### Example D: Predict a Directory

```bash
python predict.py \
  --task segment \
  --data path/to/images_dir \
  --weights run/train/exp22/weights/best.pth \
  --model-name swin_small_patch4_window7_224 \
  --num-classes 2 \
  --ann-file data/TOMATO.v5i.coco-segmentation/annotation/annotations_val.coco.json \
  --device cuda:0
```

### Example E: DETR Inference with NMS Disabled

```bash
python predict.py \
  --task detr_segment \
  --data path/to/images_dir \
  --weights run/train/exp33/weights/best.pth \
  --model-name swin_tiny_patch4_window7_224 \
  --num-classes 2 \
  --no-nms \
  --device cuda:0 \
  --draw
```

## 🔧 ONNX Export

Export a trained checkpoint to ONNX format. Task type is auto-detected from the checkpoint keys.

```bash
# Export (task auto-detected)
python onnx_tools/export_onnx.py \
  --weights run/train/expN/weights/best.pth \
  --model swin_small_patch4_window7_224 \
  --num-classes 2

# Override output path
python onnx_tools/export_onnx.py \
  --weights run/train/expN/weights/best.pth \
  --model swin_small_patch4_window7_224 \
  --num-classes 2 \
  --output exported/model.onnx

# Force task type
python onnx_tools/export_onnx.py \
  --weights best.pth \
  --model vit_base_patch16_224_in21k \
  --num-classes 26 \
  --task classify
```

By default the `.onnx` file is saved next to the weights file with the same name.

Verify the exported model matches PyTorch outputs (requires `pip install onnxruntime`):

```bash
python onnx_tools/verify_export_onnx.py \
  --weights run/train/expN/weights/best.pth \
  --model swin_small_patch4_window7_224 \
  --num-classes 2
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--weights` | required | `.pt` / `.pth` checkpoint |
| `--model` | `vit_base_patch16_224_in21k` | backbone architecture name |
| `--num-classes` | required | foreground class count |
| `--task` | `auto` | `auto` / `classify` / `detect` / `segment` |
| `--output` | same dir as weights | output `.onnx` path |
| `--input-size` | `224` | input image size |
| `--opset` | `17` | ONNX opset version |
| `--device` | `cpu` | `cpu` or `cuda` |

## 📁 Outputs

Training outputs are saved under `run/train/expN/`:

**Classification:**
- `weights/best.pth`, `weights/last.pth`
- `metrics.csv`
- `class_indices.json`
- `loss_curve.png`, `acc_curve.png`, `val_prf_curve.png`, `confusion_matrix.png`

**Detection / Segmentation (R-CNN and DETR):**
- `weights/best.pth`, `weights/last.pth`
- `metrics.csv`
- `calibration_curve.png`, `confusion_matrix.png`
- `f1_confidence.png`, `per_class_ap.png`, `pr_curve.png`
- `scale_analysis.png` (segmentation only)
- `mask_analysis/` (segmentation only)

> When resuming training, `metrics.csv` is **appended** (not overwritten), so the final plots show the complete training history from epoch 0 to the new `--epochs` target.

Prediction outputs are saved under `run/predict/expN/predictions.txt`.

## 📝 Notes

- If CUDA is unavailable, training may fall back to CPU and become much slower.
- Detection/segmentation workloads are significantly heavier than classification.
- Make sure model architecture matches pretrained weights (patch size, embed dim, depth).
- For detect/segment/detr_* inference, `--num-classes` must match the training checkpoint exactly (foreground class count only).
- DETR tasks use **AdamW** automatically; SGD is used for classify/detect/segment.
- For DETR, `last.pth` is always preferred over `best.pth` for resume, as it carries the latest full training state.

## 🔒 Final Reminder

This project is for non-commercial learning use only. Any commercial usage is prohibited.

## 🤝 Contributions Welcome

Contributions are highly appreciated. If you want to help improve this project, you are very welcome to join.

### You can contribute by

- 🐛 Reporting bugs and edge cases
- 📝 Improving README/docs and usage examples
- ⚡ Optimizing training/evaluation speed and memory usage
- 🧪 Adding tests and reproducible experiment notes
- 🌱 Improving detect/segment/classify robustness

### Suggested contribution flow

1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Open a Pull Request with a concise description

If you find this project helpful, please consider giving it a ⭐ and sharing your ideas in Issues.


---



## some tips
*This project is not perfect. If you encounter any location errors during use, please raise an issue. Those who are capable can modify it themselves. Submitting a pull request would be even better.*

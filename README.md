# Vision Transformer for Classify, Detection and Segmentation

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-Classify%20%7C%20Detect%20%7C%20Segment-2EA44F">
  <img alt="License Notice" src="https://img.shields.io/badge/Usage-Non--Commercial%20Only-8A2BE2">
</p>

<p align="center">
  <b>🌿 ViT Learning Project for Classification, Detection, and Segmentation</b>
</p>

<p align="center">
  ✨ Clean training workflow • 📊 Built-in metrics • 🧩 COCO support • 🖼️ Visualization
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

This project brings Vision Transformer (ViT) into three classic computer vision tasks — image classification, object detection, and instance segmentation — all under one unified training and inference entry point.

The backbone is a pretrained ViT (ImageNet-21k), extended with an FPN neck and task-specific heads: a Fast R-CNN style head for detection, and a Mask R-CNN style mask branch on top for segmentation. Transfer learning is the default workflow — freeze the backbone, fine-tune the head, and get results fast even on limited hardware.

Beyond training, the project puts some effort into making results actually readable: COCO-standard mAP evaluation, per-class AP charts, PR curves, F1-confidence curves, confusion matrices with background, calibration curves, and scale/mask analysis are all generated automatically at the end of training. No extra scripts needed.

## ✨ Key Features

- Multiple ViT backbones (base / large / huge, patch 14/16/32)
- Three tasks in one entry point: classification, detection, segmentation
- Detection via ViT + FPN neck + Fast R-CNN head; segmentation adds Mask R-CNN style mask branch
- COCO-format training and evaluation with pycocotools mAP (mAP@0.5, mAP@0.5:0.95)
- Classification metrics: loss, accuracy, macro precision / recall / F1 per epoch
- Rich post-training evaluation charts for detection/segmentation:
  - PR curves, F1-Confidence curve, per-class AP bar chart
  - Confusion matrix (with background class), calibration curve
  - Scale analysis (small / medium / large), mask IoU analysis (segmentation)
- Auto-increment experiment folders (`run/train/expN`)
- Inference with optional visualization (`--draw`) and TSV prediction export
### 🔍 About Detection
The traditional VIT backbone was used, a YOLO-like FPN neck was added, and finally, a torch-style Fast R-CNN was used for the final regression and classification.

### 🧩 About Segmentation
The same ViT + FPN backbone pipeline is used, and a Mask R-CNN style mask branch is added on top of the detection head to predict pixel-level instance masks together with box classification and regression.

## 🗂️ Project Structure

- `train.py`: Unified entry for `classify`, `detect`, `segment`
- `predict.py`: Inference for single image or directory
- `model/vit_model.py`: ViT backbone implementation
- `model/detection_head.py`: Detection head
- `model/segmentation_head.py`: Segmentation head
- `tools/my_dataset.py`: Classification dataset and dataloaders
- `tools/coco_dataset.py`: COCO dataset and dataloaders
- `tools/utils.py`: Training/evaluation helpers
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

### Detection / Segmentation

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

### Train Args Quick Reference

| Argument | Task | Meaning |
|----------|------|---------|
| `--task` | all | choose from classify / detect / segment |
| `--model` | train | ViT backbone variant used in training |
| `--weights` | train | pretrained or resume checkpoint path |
| `--batch-size` | train | batch size (reduce if OOM) |
| `--eval-interval` | detect/segment | run evaluation every N epochs (and always at last epoch) |
| `--freeze-layers` | classify | freeze backbone, train head/pre_logits |

### Predict Args Quick Reference

| Argument | Meaning |
|----------|---------|
| `--task` | inference mode: classify / detect / segment |
| `--data` | single image path or image directory |
| `--weights` | checkpoint used for prediction |
| `--num-classes` | required for detect/segment |
| `--ann-file` | optional COCO json for class-name mapping in detect/segment |
| `--class-indices` | class name json (mainly for classification) |
| `--draw` | save visualized predictions |

### Confidence Threshold Policy

- Detection and segmentation predictions are filtered by a global confidence threshold in `predict.py`.
- Current default in code is `CONF_THRESH = 0.9`.
- Only predictions with score >= CONF_THRESH are written to `predictions.txt` and drawn to output images.

### Label Name Mapping Priority (Detect/Segment)

When generating class names for detect/segment prediction, the mapping priority is:
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

1. Prediction has objects but wrong class names.
  - Check `--ann-file` points to the correct COCO json and verify `categories` content/order.

2. Prediction output is empty.
  - Current threshold may be too strict (`CONF_THRESH = 0.9`), especially on hard images.

3. Detect/segment training seems stuck.
  - Evaluation and data loading can take long; monitor epoch/batch logs and GPU utilization.

4. Git push fails intermittently.
  - Usually network/TLS issue; retry push after switching network or DNS.



## 🚀 Training

> Tip: If your GPU memory is limited, lower `--batch-size` first and keep `--eval-interval` larger for smoother training.

### 💡 Tips

- **Freeze the backbone first.** Unless you have a large dataset, keep `--freeze-layers True` and only fine-tune the head. It trains faster and avoids overfitting on small data.

- **VRAM is the main bottleneck.** ViT is memory-hungry. If you hit OOM, drop `--batch-size` to 2–4 first. Detection/segmentation needs at least 8 GB; classification is more forgiving.

- **Evaluation runs on the last epoch (and every `--eval-interval` epochs).** The default interval is 10 to keep training fast. Set `--eval-interval 1` if you want per-epoch mAP, but expect slower runs.

- **mAP follows the COCO standard.** Both mAP@0.5 and mAP@0.5:0.95 are computed with pycocotools, so numbers are directly comparable to other COCO benchmarks.

- **Class names come from your COCO JSON.** If labels look wrong in outputs or charts, check the `categories` field in your annotation file.



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
  --train-img-dir data/val2017 \
  --train-ann-file data/annotations_trainval2017/annotations/instances_val2017.json \
  --val-img-dir data/val2017 \
  --val-ann-file data/annotations_trainval2017/annotations/instances_val2017.json \
  --model vit_base_patch16_224_in21k \
  --weights weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth \
  --epochs 100 \
  --eval-interval 10 \
  --batch-size 8 \
  --device cuda:0
```

### 3) Segmentation

```bash
python train.py \
  --task segment \
  --train-img-dir data/val2017 \
  --train-ann-file data/annotations_trainval2017/annotations/instances_val2017.json \
  --val-img-dir data/val2017 \
  --val-ann-file data/annotations_trainval2017/annotations/instances_val2017.json \
  --model vit_base_patch16_224_in21k \
  --weights weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth \
  --epochs 100 \
  --eval-interval 10 \
  --batch-size 8 \
  --device cuda:0
```

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
  --data path/to/target.jpg \
  --weights run/train/exp/weights/best.pth \
  --class-indices run/train/exp/class_indices.json \
  --model-name vit_base_patch16_224_in21k \
  --device cuda:0 \
  --draw
```

### Example D: Predict a Directory

```bash
python predict.py \
  --data path/to/images_dir \
  --weights run/train/exp/weights/best.pth \
  --class-indices run/train/exp/class_indices.json \
  --model-name vit_base_patch16_224_in21k \
  --device cuda:0
```

## 📁 Outputs

Training outputs are saved under `run/train/expN/`,   

classify including:
- `weights/best.pth`
- `weights/last.pth`
- `metrics.csv`
- `class_indices.json`
- `loss_curve.png`
- `acc_curve.png`
- `val_prf_curve.png`
- `confusion_matrix.png`

Prediction outputs are saved under `run/predict/expN/predictions.txt`.  

Detection/segmentation including:  
- `weights/best.pth`
- `weights/last.pth`
- `metrics.csv`
- `calibration_curve.png`
- `confusion_matrix.png`
- `f1_confidence.png`
- `mask_analysis`
- `per_class_ap.png`
- `pr_curve.png`
- `scale_analysis.png`(Segmentation unique)



## 📝 Notes

- If CUDA is unavailable, training may fall back to CPU and become much slower.
- Detection/segmentation workloads are significantly heavier than classification.
- Make sure model architecture matches pretrained weights (patch size, embed dim, depth).

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

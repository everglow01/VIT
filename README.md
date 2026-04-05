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

This project is a Vision Transformer (ViT)-based computer vision learning project with three supported tasks:
- Image classification 
- Object detection (COCO format)
- Instance segmentation (COCO format)

It supports transfer learning from ImageNet-21k pretrained ViT weights and includes training, evaluation, inference, and visualization utilities.

## ✨ Key Features

- Multiple ViT backbones (base / large / huge)
- Classification metrics (loss, accuracy, macro precision/recall/F1)
- Detection and segmentation training pipeline
- Auto-increment experiment folders (`run/train/expN`)
- Metric plotting and confusion matrix generation
- Prediction export for inference runs
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

### 💡 tips
1.For datasets with small amounts of data or small GPUs, it is recommended to freeze the backbone and use a pre-trained backbone.  

2.This model has high VRAM requirements. The default batch size may result in a "CUDA is out of memory" error. At least 8GB of VRAM is required, and batch sizes of 2-4 are recommended.  

3.The default evaluation method produces the same results when processing mAP50 and mAP50_90. If you need to switch to the native COCO dataset evaluation mode, you will need to manually modify the settings.(The issue has been fixed; all calculations have now been changed to conform to the COCO standard for mAP calculation.)  

4.The evaluation mode defaults to evaluating every ten rounds (or the last round). This was modified on my computer to reduce the reasoning burden. If needed, you can change the default to evaluating every round.  

5.The default label names come from your COCO dataset's JSON file. If the label names are incorrect, please check if your JSON file format and content are correct.   



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

Training outputs are saved under `run/train/expN/`, including:
- `weights/best.pth`
- `weights/last.pth`
- `metrics.csv`
- `class_indices.json`
- `loss_curve.png`
- `acc_curve.png`
- `val_prf_curve.png`
- `confusion_matrix.png`

Prediction outputs are saved under `run/predict/expN/predictions.txt`.



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

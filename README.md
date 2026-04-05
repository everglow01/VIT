# Vision Transformer for Classify, Detection and Segmentation

## Important Notice (Non-Commercial Only)

This repository is provided strictly for learning, research, and educational demonstration.

Commercial use is strictly prohibited in any form, including but not limited to:
- Selling this codebase or trained models
- Integrating this project into paid products or services
- Using this project in profit-oriented production environments

By using this repository, you agree to these restrictions.

## Overview

This project is a Vision Transformer (ViT)-based computer vision learning project with three supported tasks:
- Image classification 
- Object detection (COCO format)
- Instance segmentation (COCO format)

It supports transfer learning from ImageNet-21k pretrained ViT weights and includes training, evaluation, inference, and visualization utilities.

## Key Features

- Multiple ViT backbones (base / large / huge)
- Classification metrics (loss, accuracy, macro precision/recall/F1)
- Detection and segmentation training pipeline
- Auto-increment experiment folders (`run/train/expN`)
- Metric plotting and confusion matrix generation
- Prediction export for inference runs
### About Detection
The traditional VIT backbone was used, a YOLO-like FPN neck was added, and finally, a torch-style Fast R-CNN was used for the final regression and classification.

### About Segmentation
The same ViT + FPN backbone pipeline is used, and a Mask R-CNN style mask branch is added on top of the detection head to predict pixel-level instance masks together with box classification and regression.

## Project Structure

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

## Installation

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

## Data Preparation

### Classification

Use ImageFolder-style data (one class per subfolder).

### Detection / Segmentation

Use COCO-format data:
- Image directory (for example `data/val2017`)
- Annotation file (for example `instances_val2017.json`)

## Training

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

## Usage Examples

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

## Outputs

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

## Notes

- If CUDA is unavailable, training may fall back to CPU and become much slower.
- Detection/segmentation workloads are significantly heavier than classification.
- Make sure model architecture matches pretrained weights (patch size, embed dim, depth).

## Final Reminder

This project is for non-commercial learning use only. Any commercial usage is prohibited.

## some tips
*This project is not perfect. If you encounter any location errors during use, please raise an issue. Those who are capable can modify it themselves. Submitting a pull request would be even better.*
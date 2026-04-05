# Vision Transformer - Plant Leaf Disease Classification

## Project Overview
Transfer learning project using ViT-Base (ImageNet-21k pre-trained) to classify 26 plant leaf disease classes across 63,545 images.

## Key Files
- `model/vit_model.py` — ViT architecture (PatchEmbed, Attention, Block, VisionTransformer)
- `train.py` — Training script with CLI args
- `predict.py` — Inference script (single image or directory)
- `tools/my_dataset.py` — Dataset class and DataLoader builder
- `tools/utils.py` — Training loop, evaluation, metrics
- `tools/plot_metrics.py` — Loss/accuracy/PRF curves, confusion matrix
- `tools/create_exp_folder.py` — Auto-increments run/train/expN folders

## Setup
```bash
pip install -r requirements.txt
```
Dependencies: torch, torchvision, tqdm, Pillow, numpy, pandas, matplotlib

## Training
```bash
python train.py \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.001 \
  --lrf 0.01 \
  --data-path Plant_Leaf_Disease \
  --model vit_base_patch16_224_in21k \
  --weights weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth \
  --freeze-layers True \
  --device cuda:0
```

Outputs to `run/train/expN/`:
- `weights/best.pth`, `weights/last.pth`
- `metrics.csv` (loss, acc, macro P/R/F1 per epoch)
- `class_indices.json`
- `loss_curve.png`, `acc_curve.png`, `val_prf_curve.png`, `confusion_matrix.png`

## Inference
```bash
python predict.py \
  --data <image_or_dir> \
  --weights run/train/exp/weights/best.pth \
  --class-indices run/train/exp/class_indices.json \
  --model-name vit_base_patch16_224_in21k \
  --device cuda:0 \
  --draw
```

Outputs to `run/predict/expN/predictions.txt` (TSV: path, pred_id, pred_name, probability)

## Model Variants
| Name | Dim | Blocks | Patch |
|------|-----|--------|-------|
| `vit_base_patch16_224_in21k` | 768 | 12 | 16×16 |
| `vit_base_patch32_224_in21k` | 768 | 12 | 32×32 |
| `vit_large_patch16_224_in21k` | 1024 | 24 | 16×16 |
| `vit_large_patch32_224_in21k` | 1024 | 24 | 32×32 |
| `vit_huge_patch14_224_in21k` | 1280 | 32 | 14×14 |

## Data
- Path: `Plant_Leaf_Disease/` — 26 classes, 63,545 images
- Split: 80% train / 20% val (stratified)
- Input: 224×224 RGB, normalized mean/std=0.5
- Train augmentation: RandomResizedCrop + RandomHorizontalFlip
- Val: Resize(256) → CenterCrop(224)

## Training Details
- Optimizer: SGD (momentum=0.9, weight_decay=5e-5)
- LR schedule: Cosine annealing (lr → lr×lrf)
- Default: freeze backbone, train head + pre_logits only
- Loss: CrossEntropyLoss
- Metrics: macro Precision, Recall, F1 (from confusion matrix)

## Pre-trained Weights
`weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth` — 412MB, ImageNet-21k

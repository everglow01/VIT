"""
COCO-format dataset for detection and instance segmentation tasks.
Only imported when --task detect or --task segment is used.

Expected data layout (user provides paths explicitly):
  --train-img-dir  data/train2017/
  --train-ann-file data/annotations/instances_train2017.json
  --val-img-dir    data/val2017/
  --val-ann-file   data/annotations/instances_val2017.json
"""
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class CocoDataset(Dataset):
    """
    COCO-format dataset returning (image_tensor, target) pairs.

    target dict keys:
      boxes   : FloatTensor [N, 4]  (x1, y1, x2, y2)
      labels  : Int64Tensor [N]
      image_id: Int64Tensor [1]
      area    : FloatTensor [N]
      iscrowd : Int64Tensor [N]
      masks   : UInt8Tensor [N, H, W]  (only when load_masks=True)
    """
    def __init__(self, img_dir: str, ann_file: str, load_masks: bool = False, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.load_masks = load_masks
        self.transforms = transforms

        # Only keep images that have at least one annotation
        self.ids = sorted([
            img_id for img_id in self.coco.imgs
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
        ])

        # Build contiguous label mapping: COCO category ids -> 1-based contiguous ids
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowds = [], [], [], []
        masks = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(ann["area"])
            iscrowds.append(ann.get("iscrowd", 0))

            if self.load_masks:
                m = self.coco.annToMask(ann)  # numpy [H, W] uint8
                masks.append(m)

        target = {
            "boxes":    torch.as_tensor(boxes,    dtype=torch.float32).reshape(-1, 4),
            "labels":   torch.as_tensor(labels,   dtype=torch.int64),
            "image_id": torch.tensor([img_id],    dtype=torch.int64),
            "area":     torch.as_tensor(areas,    dtype=torch.float32),
            "iscrowd":  torch.as_tensor(iscrowds, dtype=torch.int64),
        }
        if self.load_masks:
            if masks:
                target["masks"] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, H, W), dtype=torch.uint8)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def _coco_collate_fn(batch):
    """Collate variable-length targets into a list (required by torchvision detection models)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def build_coco_dataloaders(
    train_img_dir: str,
    train_ann_file: str,
    val_img_dir: str,
    val_ann_file: str,
    batch_size: int = 4,
    load_masks: bool = False,
    num_workers: int = 4,
):
    """
    Returns (train_loader, val_loader, num_classes).
    Images are converted to tensors; normalization is handled inside the model transform.
    """
    to_tensor = T.ToTensor()

    train_ds = CocoDataset(train_img_dir, train_ann_file, load_masks=load_masks, transforms=to_tensor)
    val_ds   = CocoDataset(val_img_dir,   val_ann_file,   load_masks=load_masks, transforms=to_tensor)

    assert train_ds.num_classes == val_ds.num_classes, \
        "Train and val annotation files have different category sets."

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_coco_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=_coco_collate_fn, pin_memory=True,
    )
    return train_loader, val_loader, train_ds.num_classes

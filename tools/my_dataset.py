from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Tuple, Any, Optional
import os
import torch
from torchvision import transforms


class ViTDataSet(Dataset):
    def __init__(self,
                 images_path: Sequence[str],
                 images_class: Sequence[int],
                 transform: Optional[Any] = None):
        assert len(images_path) == len(images_class), \
            f"images_path and images_class length mismatch: {len(images_path)} vs {len(images_class)}"

        self.images_path = list(images_path)
        self.images_class = list(images_class)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images_path[idx]
        label = int(self.images_class[idx])

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return images, labels


def build_vit_dataloaders(train_images_path, train_images_label,
                          val_images_path, val_images_label,
                          batch_size, img_size=224,
                          mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                          num_workers=None):
    """
    Build train/val DataLoaders for ViT classification.

    Args:
        num_workers: worker count for data loading; None = auto (min of cpu_count, batch_size, 8)
    """
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    train_dataset = ViTDataSet(
        train_images_path, train_images_label,
        transform=data_transform["train"])
    val_dataset = ViTDataSet(
        val_images_path, val_images_label,
        transform=data_transform["val"])

    if num_workers is None:
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn
    )

    return train_loader, val_loader

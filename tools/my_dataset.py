from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Tuple, Any, Optional
import os
import torch
from torchvision import transforms


class ViTDataSet(Dataset):
    """
    自定义分类数据集：返回 (image_tensor, label)
    - image_tensor: Tensor[3, H, W]（经过 transform 后，通常 H=W=224）
    - label: int（类别 id）
    """

    def __init__(self,
                 images_path: Sequence[str],
                 images_class: Sequence[int],
                 transform: Optional[Any] = None):
        """
        Args:
            images_path: 图片路径列表/序列，例如 [".../classA/1.jpg", ".../classB/2.png", ...]
            images_class: 每张图片对应的类别 id 列表/序列，例如 [0, 0, 2, 4, ...]
            transform: torchvision.transforms 的组合，用于图像预处理/增强（PIL -> Tensor）
        """
        # 数据一致性检查：路径数与标签数必须一致
        assert len(images_path) == len(images_class), \
            f"images_path and images_class length mismatch: {len(images_path)} vs {len(images_class)}"

        self.images_path = list(images_path)
        self.images_class = list(images_class)
        self.transform = transform

    def __len__(self) -> int:
        """返回数据集样本数（DataLoader 用它确定 epoch 的长度）"""
        return len(self.images_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        根据索引 idx 取出一条样本：
        1) 读取图片（PIL）
        2) 强制转为 RGB（保证输出 3 通道；灰度图会复制到 RGB 三通道）
        3) 取标签
        4) 做 transform（RandomCrop/ToTensor/Normalize 等）
        5) 返回 (img_tensor, label)
        """
        img_path = self.images_path[idx]
        label = int(self.images_class[idx])

        # 用 with 确保文件句柄及时释放（更稳健）
        with Image.open(img_path) as img:
            # 强制转换为 RGB：
            #    - 灰度图(L) -> RGB：每个像素 (g,g,g)
            #    - RGBA -> RGB：丢弃 alpha
            #    - CMYK 等 -> RGB：统一到 3 通道
            img = img.convert("RGB")

            # transform 通常会把 PIL 图像转为 Tensor[3,H,W] 并做归一化/增强
            if self.transform is not None:
                img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自定义 batch 组装函数（给 DataLoader 使用）

        batch: list/tuple，长度=batch_size
               每个元素是 (img_tensor, label)

        返回：
          - images: Tensor[B, 3, H, W]
          - labels: Tensor[B]（dtype=torch.long，适配 CrossEntropyLoss）
        """
        # 将 [(img1,l1),(img2,l2),...] 拆成 ([img1,img2,...], [l1,l2,...])
        images, labels = tuple(zip(*batch))

        # stack 要求每张图尺寸一致（你的 transform 已裁剪到固定 img_size，因此成立）
        images = torch.stack(images, dim=0)

        # CrossEntropyLoss 需要 labels 是 int64(LongTensor)
        labels = torch.as_tensor(labels, dtype=torch.long)

        return images, labels

def build_vit_dataloaders(train_images_path, train_images_label,
                          val_images_path, val_images_label,
                          batch_size, img_size=224,
                          mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                          num_workers=None):
    """
    构建 ViT 训练/验证的 DataLoader。

    输入：
      - train_images_path / val_images_path: 图片文件路径列表（list[str]）
      - train_images_label / val_images_label: 对应标签列表（list[int]）
      - batch_size: 每个 batch 的样本数
      - img_size: 模型输入分辨率（ViT-B/16 默认224）
      - mean/std: Normalize的均值/方差（这里是把 [0,1] 线性映射到近似 [-1,1]）
      - num_workers: DataLoader 多进程加载数据的 worker 数，None 则自动估算

    输出：
      - train_loader, val_loader
    """

    # 定义训练/验证的数据预处理与增强（torchvision.transforms）
    #    - train：随机裁剪 + 翻转（增强，提高泛化）
    #    - val：固定resize + center crop（保证评估稳定可复现）
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),   # 随机裁剪到 img_size，并随机缩放
            transforms.RandomHorizontalFlip(),        # 随机水平翻转
            transforms.ToTensor(),                    # PIL -> Tensor，范围变为 [0,1]
            transforms.Normalize(mean, std),          # 归一化（常见设置 mean=std=0.5）
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),                   # 先把短边缩放到 256（经典 ImageNet eval 流程）
            transforms.CenterCrop(img_size),          # 再中心裁剪到 img_size
            transforms.ToTensor(),                    # PIL -> Tensor，[0,1]
            transforms.Normalize(mean, std),          # 同样归一化
        ]),
    }

    # 用自定义Dataset封装“读图 + label + transform”
    # Dataset 的 __getitem__ 返回：(img_tensor, label)
    train_dataset = ViTDataSet(
        train_images_path, train_images_label,
        transform=data_transform["train"])
    val_dataset = ViTDataSet(
        val_images_path, val_images_label,
        transform=data_transform["val"])

    # 自动设置num_workers（加载数据的进程数）
    #  经验策略：不超过 CPU 核数、不超过 8，也不超过 batch_size（避免过多进程反而开销大）
    if num_workers is None:
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 训练集 DataLoader
    # - shuffle=True：每个 epoch 打乱数据顺序
    # - pin_memory=True：若用GPU，可加速 CPU->GPU 拷贝
    # - collate_fn：自定义 batch 组装方式（通常用于处理不同尺寸/额外信息）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )

    # 验证集 DataLoader
    # - shuffle=False：验证/测试不需要打乱，保证评估稳定
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn
    )

    # 返回两个loader，供训练循环使用
    return train_loader, val_loader


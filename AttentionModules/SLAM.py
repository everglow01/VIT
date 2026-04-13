"""
EN:
SLAM is a lightweight composite attention block that jointly models
height-aware, width-aware, and spatial attention, then fuses them by
element-wise multiplication.
It is conceptually similar to axis-aware attention (e.g., triplet-style ideas),
but uses simple pooling + lightweight gates for each branch.

High-level structure:
1) Height-aware branch (H branch)
     - For input X: (B, C, H, W), aggregate along height dimension:
         avg_h = mean(X, dim=H), max_h = max(X, dim=H), both shaped (B, C, 1, W).
     - Fuse descriptors by addition and pass through a shared channel gate
         (1x1 bottleneck conv stack) + sigmoid to get M_h: (B, C, 1, W).

2) Width-aware branch (W branch)
     - Aggregate along width dimension:
         avg_w, max_w -> (B, C, H, 1).
     - Use the same channel gate + sigmoid to obtain M_w: (B, C, H, 1).

3) Spatial branch (S branch)
     - Aggregate across channel dimension:
         avg_s, max_s -> (B, 1, H, W).
     - Fuse and process with spatial convolution (kxk) + sigmoid to get
         M_s: (B, 1, H, W).

4) Fusion/output
     - Broadcast multiply all gates with input:
         Out = X * M_h * M_w * M_s.
     - Output shape remains (B, C, H, W).

Why this design:
- Captures directional cues separately for height and width.
- Adds spatial saliency modeling with minimal extra cost.
- Multiplicative fusion enforces consensus among multiple attention views.

ZH:
SLAM 是一个轻量复合注意力模块，同时建模“高度方向注意力、宽度方向注意力、
空间注意力”，并通过逐元素乘法进行融合。
它与轴向感知注意力（如 triplet 类思路）相近，但采用了更简洁的
池化 + 轻量门控实现。

整体结构：
1）高度方向分支（H 分支）
     - 对输入 X（B, C, H, W）在高度维做聚合：
         avg_h = mean(X, dim=H)、max_h = max(X, dim=H)，形状均为 (B, C, 1, W)。
     - 两者相加后通过共享通道门控（1x1 瓶颈卷积堆叠）和 sigmoid，
         得到 M_h，形状为 (B, C, 1, W)。

2）宽度方向分支（W 分支）
     - 在宽度维做聚合，得到 avg_w、max_w，形状为 (B, C, H, 1)。
     - 通过同一通道门控 + sigmoid，得到 M_w，形状为 (B, C, H, 1)。

3）空间分支（S 分支）
     - 在通道维做聚合，得到 avg_s、max_s，形状为 (B, 1, H, W)。
     - 融合后经空间卷积（kxk）和 sigmoid，得到 M_s，形状为 (B, 1, H, W)。

4）融合与输出
     - 将三个注意力图与输入做广播逐元素相乘：
         Out = X * M_h * M_w * M_s。
     - 输出形状保持为 (B, C, H, W)。

设计动机：
- 分别建模 H/W 方向可增强方向性上下文感知。
- 额外加入空间显著性建模且计算开销较小。
- 乘法融合让多视角注意力共同约束最终响应。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SLAM(nn.Module): # 类似TripleAttention，融合了空间、通道、局部注意力机制
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)
        
        self.hw = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        
        pad = kernel_size // 2
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        fh_avg = x.mean(dim=2, keepdim=True) # (b,c,1,w)
        fh_max = x.max(dim=2, keepdim=True)[0] # (b,c,1,w)
        fh = fh_avg + fh_max # (b,c,1,w)
        mh = self.hw(fh) # (b,c,1,w)
        mh = self.sigmoid(mh) # (b,c,1,w)

        fw_avg = x.mean(dim=3, keepdim=True) # (b,c,h,1)
        fw_max = x.max(dim=3, keepdim=True)[0] # (b,c,h,1)
        fw = fw_avg + fw_max # (b,c,h,1)
        mw = self.hw(fw) # (b,c,h,1)
        mw = self.sigmoid(mw) # (b,c,h,1)
        
        fs_avg = x.mean(dim=1, keepdim=True) # (b,1,h,w)
        fs_max = x.max(dim=1, keepdim=True)[0] # (b,1,h,w)
        fs = fs_avg + fs_max # (b,1,h,w)
        ms = self.spatial_conv(fs) # (b,1,h,w)
        ms = self.sigmoid(ms) # (b,1,h,w)
        
        out = x * mh * mw * ms # 广播到同形状
        return out
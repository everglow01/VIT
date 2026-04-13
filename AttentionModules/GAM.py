"""
EN:
GAM (Global Attention Mechanism) refines features with two sequential stages:
channel attention followed by spatial attention.
Unlike channel attention that only pools globally, this implementation applies
an MLP over per-location channel vectors, then applies a spatial convolutional
attention map on the channel-refined features.

High-level structure:
1) Channel attention stage
    - Input X: (B, C, H, W), let N = H * W.
    - Rearrange to (B, N, C), treating each spatial position as a C-dim token.
    - Apply channel MLP (C -> hidden -> C) to each token independently.
    - Sigmoid produces channel gates M_c, then reshape back to (B, C, H, W).
    - Reweight feature: X_c = X * M_c.

2) Spatial attention stage
    - On X_c, apply a spatial subnetwork:
      1x1 Conv (C -> hidden) + BN + ReLU,
      then kxk Conv (hidden -> C) + BN.
    - Sigmoid produces spatially-aware map M_s: (B, C, H, W).
    - Reweight again: Out = X_c * M_s.

3) Output
    - Final output keeps the same shape as input: (B, C, H, W).

Why this design:
- Channel stage models inter-channel dependencies at each spatial location.
- Spatial stage further emphasizes informative regions after channel refinement.
- Sequential channel+spatial gating improves feature discrimination with moderate cost.

ZH:
GAM（Global Attention Mechanism，全局注意力机制）采用串联的两阶段增强：
先通道注意力，再空间注意力。
与仅做全局池化的通道注意力不同，这个实现先在每个空间位置的通道向量上做 MLP，
再对通道增强后的特征施加卷积式空间注意力。

整体结构：
1）通道注意力阶段
    - 输入 X 形状为 (B, C, H, W)，令 N = H * W。
    - 将特征重排为 (B, N, C)，把每个空间位置看作一个 C 维 token。
    - 对每个 token 独立应用通道 MLP（C -> hidden -> C）。
    - 经过 sigmoid 得到通道门控 M_c，再还原回 (B, C, H, W)。
    - 完成第一次重标定：X_c = X * M_c。

2）空间注意力阶段
    - 在 X_c 上应用空间子网络：
      1x1 卷积（C -> hidden）+ BN + ReLU，
      再接 kxk 卷积（hidden -> C）+ BN。
    - 经 sigmoid 得到空间感知权重 M_s，形状为 (B, C, H, W)。
    - 完成第二次重标定：Out = X_c * M_s。

3）输出
    - 最终输出与输入形状一致： (B, C, H, W)。

设计动机：
- 通道阶段在每个空间位置上建模通道依赖关系。
- 空间阶段在通道增强后进一步突出关键区域。
- 串联门控在较适中的开销下提升特征判别能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAM(nn.Module):
    def __init__(self, channels, reduction=16, L=1, spatial_kernal=7):
        super().__init__()
        assert channels > 0
        hidden = max(L, channels//reduction)
        
        self.channels_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        
        self.channels_sigmoid = nn.Sigmoid()
        
        pad = spatial_kernal // 2 # 确保卷积后空间维不变
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=spatial_kernal, padding=pad, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.spatial_sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C,H, W = x.shape
        N = H * W
        
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, N, C) # (B, N, C)
        mc = self.channels_mlp(x_reshaped) # (B, N, C)
        mc = self.channels_sigmoid(mc).view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        x_c = x * mc
        
        ms = self.spatial(x_c) # (B, C, H, W)
        ms = self.spatial_sigmoid(ms)
        out = x_c * ms
        
        return out
"""
EN:
BAM (Bottleneck Attention Module) is a lightweight attention block that combines
channel attention and spatial attention in parallel, then fuses them into a single
attention map for feature reweighting.

High-level structure:
1) Channel branch
    - Global average pooling compresses spatial dimensions (H, W) to 1x1.
    - A bottleneck 1x1 conv stack (C -> C/r -> C) models inter-channel dependency.
    - Output shape is (N, C, 1, 1), which is broadcastable to the input feature map.

2) Spatial branch
    - A 1x1 conv first reduces channels (C -> C/r).
    - Multiple 3x3 dilated convolutions capture multi-scale context with larger
      receptive fields while keeping spatial resolution.
    - A final 1x1 conv projects features to a single-channel spatial map
      with shape (N, 1, H, W).

3) Fusion and reweighting
    - Channel map and spatial map are added (with broadcasting) and passed through
      sigmoid to get attention M in [0, 1].
    - The module returns x * (1 + M), i.e., residual-style amplification so that
      important regions/channels are emphasized while preserving the original signal path.

Why this design:
- Bottleneck design controls parameter/computation cost.
- Dilated convolutions expand context aggregation ability.
- Residual reweighting improves stability versus pure multiplicative gating.

ZH:
BAM（瓶颈注意力模块）是一种轻量注意力结构，将通道注意力与空间注意力并行建模，
再融合为一个统一的注意力图，对输入特征进行重标定。

整体结构：
1）通道分支
    - 通过全局平均池化将空间维度 (H, W) 压缩到 1x1。
    - 使用瓶颈式 1x1 卷积堆叠（C -> C/r -> C）学习通道间依赖关系。
    - 输出张量形状为 (N, C, 1, 1)，可广播到原特征图。

2）空间分支
    - 先用 1x1 卷积降维（C -> C/r）。
    - 通过多层 3x3 空洞卷积提取多尺度上下文，在保持分辨率的同时扩大感受野。
    - 最后用 1x1 卷积映射为单通道空间注意力图，形状为 (N, 1, H, W)。

3）融合与重标定
    - 通道图与空间图相加（依赖广播机制）后经过 sigmoid，得到 [0, 1] 的注意力 M。
    - 最终输出为 x * (1 + M)，即残差式增强：在保留原始信息通路的前提下强化关键特征。

设计动机：
- 瓶颈结构可降低参数量与计算量。
- 空洞卷积有助于聚合更大范围上下文信息。
- 残差式门控通常比纯乘法门控更稳定。
"""

import torch
import torch.nn as nn
class BAM(nn.Module):
    '''Bottleneck Attention Module (BAM) for enhanced feature representation.'''
    '''sturcture: channel attention + spatial attention'''
    '''BAM:增强特征表示的瓶颈注意力模块。'''
    '''结构：通道注意力 + 空间注意力'''
    

    def __init__(self, channels, reduction=16, dilations=(1,2,4)):# channels: 输入特征图的通道数, reduction: 隐藏层通道数的缩放因子(分母), dilations: 空洞卷积的膨胀率列表
        super().__init__()

        assert channels > 0
        hidden = max(1, channels//reduction)
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        
        self.Conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        
        self.SpatialDilation = nn.Sequential(
            nn.Conv2d(channels, hidden,1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        
        convs = []
        for d in dilations: # 空洞卷积扩大感受野 适合捕捉大尺度或被遮挡的目标
            convs += [
                nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d, bias=False),# 空洞卷积，padding=d保持输出尺寸不变
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True)
            ]
        self.SpatialConvs = nn.Sequential(*convs)
        self.spatial_out = nn.Conv2d(hidden, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        M_c = self.GlobalAvgPool(x)
        M_c = self.Conv(M_c)
        
        s = self.SpatialDilation(x)
        s = self.SpatialConvs(s)
        M_s = self.spatial_out(s)
        M = self.sigmoid(M_c + M_s)
        
        return x * (1.0 + M) # 广播到同形状 残差链接
            

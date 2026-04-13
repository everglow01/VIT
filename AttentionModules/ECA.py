"""
EN:
ECA (Efficient Channel Attention) is a lightweight channel attention mechanism
that avoids dimensionality reduction and models local cross-channel interaction
with a 1D convolution on channel descriptors.

Core idea:
- Start from global channel descriptors via GAP.
- Replace SE-style bottleneck MLP with a local 1D convolution.
- Use adaptive kernel size to match channel dimension, balancing capacity and cost.

High-level structure:
1) Channel descriptor extraction
     - Input X: (B, C, H, W).
     - Global average pooling produces Y_g: (B, C, 1, 1).

2) Local cross-channel interaction
     - Reshape Y_g to a 1D sequence over channels: typically (B, 1, C).
     - Apply Conv1d(1 -> 1, kernel_size = k, padding = (k-1)/2) to capture
         local dependencies among neighboring channels.

3) Attention generation and scaling
     - Sigmoid maps channel responses to [0, 1].
     - Reshape back to (B, C, 1, 1) and multiply with input by broadcasting:
         Out = X * A_c.

Adaptive kernel-size rule:
- When k is not manually specified, ECA uses a channel-dependent odd kernel:
    k = odd(|log2(C)/gamma + b|), with a minimum of 3 in this implementation.
- This allows larger channel dimensions to use slightly wider local interaction.

Why this design:
- Very low parameter/computation overhead.
- Preserves channel dimension without compression loss.
- Often provides strong accuracy-efficiency tradeoff in CNN backbones.

ZH:
ECA（Efficient Channel Attention，高效通道注意力）是一种轻量通道注意力机制，
它不做通道降维，而是通过对通道描述做 1D 卷积来建模局部通道交互。

核心思想：
- 先通过全局池化得到通道描述；
- 用局部 1D 卷积替代 SE 中的瓶颈 MLP；
- 根据通道数自适应选择卷积核大小，在表达能力与开销之间取得平衡。

整体结构：
1）通道描述提取
     - 输入 X 形状为 (B, C, H, W)。
     - 全局平均池化后得到 Y_g，形状为 (B, C, 1, 1)。

2）局部通道交互建模
     - 将 Y_g 重排为沿通道展开的一维序列（通常为 (B, 1, C)）。
     - 通过 Conv1d(1 -> 1, kernel_size = k, padding = (k-1)/2)
         学习相邻通道之间的局部依赖关系。

3）注意力生成与重标定
     - 经过 sigmoid 得到 [0, 1] 的通道权重。
     - 再重排回 (B, C, 1, 1)，通过广播与输入逐元素相乘：
         Out = X * A_c。

自适应卷积核规则：
- 当未手动指定 k 时，按通道数计算奇数卷积核：
    k = odd(|log2(C)/gamma + b|)，本实现中最小为 3。
- 通道数越大，通常可获得更宽的局部交互范围。

设计动机：
- 参数量与计算量都很低。
- 不降维，减少信息压缩带来的损失。
- 在精度与效率之间通常有较好的平衡。
"""

import torch
import torch.nn as nn
import math
class ECA(nn.Module):
    def __init__(self, channels, kernal_size=None, gamma=2, b=1):
        super().__init__()
        assert channels > 0
        
        if kernal_size is None:
            # k = |log2(C)/gamma + b|_odd
            t = int(abs((math.log2(channels) / gamma) + b))
            kernal_size = t if t % 2 else t+1 # 确保核大小为奇数
            kernal_size = max(3, kernal_size)
            
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.Conv1d = nn.Conv1d(1, 1, kernel_size=kernal_size, padding=(kernal_size-1)//2, bias=False)
            self.sigmoid = nn.Sigmoid()
             
    def forward(self, x):
        b,c,h,w = x.shape
        y = self.gap(x)
            
        y = y.squeeze(-1).transpose(-1) # (B,C)
        y = y.unsqueeze(1) # (B,1,C)
        y = self.Conv1d(y) # (B,1,C)
            
        y = self.sigmoid(y) # (B,1,C)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1) # (B,C,1,1)
            
        # 广播
        return x*y
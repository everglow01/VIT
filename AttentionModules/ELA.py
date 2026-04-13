"""
EN:
ELA (Efficient Local/Axis Attention style block) refines features by generating
separate attention maps along height and width axes, then applying multiplicative
reweighting to the input feature map.

Core idea:
- Normalize input features first (GroupNorm) for more stable statistics.
- Build two directional descriptors with axis-wise average pooling.
- Produce height-aware and width-aware attention independently.
- Fuse by element-wise multiplication, with optional residual connection.

High-level structure:
1) Normalization
     - Input X: (B, C, H, W).
     - Apply GroupNorm to get X_gn with the same shape.
     - Group number is adjusted to be divisible by C for robustness.

2) Directional descriptor extraction
     - Height descriptor: average over width -> X_h: (B, C, H, 1).
     - Width descriptor: average over height -> X_w: (B, C, 1, W).

3) Axis attention generation
     - Squeeze singleton dimensions to 1D axis forms.
     - Pass through convolutional transforms and sigmoid to obtain:
         A_h (height attention) and A_w (width attention).
     - Reshape back to broadcastable forms:
         A_h: (B, C, H, 1), A_w: (B, C, 1, W).

4) Reweighting and optional residual
     - Apply broadcast multiplication:
         Out = X * A_h * A_w.
     - If residual is enabled:
         Out = Out + X.

Why this design:
- Axis-wise modeling captures long-range directional context efficiently.
- GroupNorm improves behavior when batch size is small.
- Multiplicative fusion remains lightweight and easy to integrate.
- Optional residual helps preserve original signal and stabilize training.

ZH:
ELA（高效局部/轴向注意力风格模块）通过分别生成“高度方向注意力”和
“宽度方向注意力”来重标定输入特征，并用逐元素乘法完成融合。

核心思想：
- 先对输入做归一化（GroupNorm），稳定特征分布；
- 再通过轴向平均池化提取两个方向描述；
- 独立生成 H/W 两个方向的注意力图；
- 最后乘法融合，并可选残差连接。

整体结构：
1）归一化阶段
     - 输入 X 形状为 (B, C, H, W)。
     - 经 GroupNorm 得到同形状特征 X_gn。
     - 分组数会调整为能整除通道数的值，提高鲁棒性。

2）方向描述提取
     - 高度描述：沿宽度维平均，得到 X_h: (B, C, H, 1)。
     - 宽度描述：沿高度维平均，得到 X_w: (B, C, 1, W)。

3）轴向注意力生成
     - 压缩单例维形成一维轴向表示。
     - 经过卷积变换与 sigmoid，得到：
         A_h（高度注意力）和 A_w（宽度注意力）。
     - 再恢复为可广播形状：
         A_h: (B, C, H, 1), A_w: (B, C, 1, W)。

4）重标定与可选残差
     - 通过广播逐元素相乘：
         Out = X * A_h * A_w。
     - 若启用残差：
         Out = Out + X。

设计动机：
- 轴向建模能高效捕捉长程方向上下文。
- GroupNorm 在小 batch 场景更稳定。
- 乘法融合轻量且易于插入现有网络。
- 可选残差有助于保留原始信息并稳定训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible_groups(channels, groups):
    if channels % groups == 0:
        return groups
    for g in range(groups, 0, -1):
        if channels % g == 0:
            return g
    return 1

class ELA(nn.Module):
    def __init__(self, channels, k=7,gn_groups=8, use_resirual = False):
        super().__init__()
        self.channels = channels
        assert k % 2 == 1, "Kernel size must be odd for 'same' padding"
        
        self.k = k
        self.use_residual = use_resirual
        
        g = _make_divisible_groups(channels, gn_groups)
        
        self.gn = nn.GroupNorm(g, channels)
        
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=1, bias=True)
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=1, bias=True)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_gn = self.gn(x)
        
        x_h = x_gn.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        x_w = x_gn.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        
        x_h_1d = x_h.squeeze(3)  # (B, C, H)
        x_w_1d = x_w.squeeze(2)  # (B, C, W)
        
        attn_h = self.sigmoid(self.conv_h(x_h_1d))
        attn_w = self.sigmoid(self.conv_w(x_w_1d))
        
        attn_h = attn_h.unsqueeze(3)  # (B, C, H, 1)
        attn_w = attn_w.unsqueeze(2)  # (B, C, 1, W)
        
        out = x * attn_h * attn_w
        if self.use_residual:
            out = out + x
        return out
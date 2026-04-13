"""
EN:
A2 module (Double Attention) performs two-stage attention aggregation to capture
global context and redistribute it back to spatial locations.
It can be viewed as: "gather global descriptors" -> "distribute descriptors".

High-level structure:
1) Input projection
     - For input X with shape (B, C, H, W), use three 1x1 conv projections:
         A = ConvA(X), Bf = ConvB(X), Cq = ConvC(X), each with shape (B, hidden, H, W).
     - Flatten spatial dimensions to N = H*W:
         A, Bf, Cq -> (B, hidden, N).

2) First attention (feature gathering)
     - Apply softmax on A along spatial dimension N to get attention weights.
     - Compute gathered features with batch matrix multiplication:
         G = Bf @ A^T, shape (B, hidden, hidden).
     - This step aggregates spatial information into compact global descriptors.

3) Second attention (feature distribution)
     - Apply softmax on Cq along spatial dimension N to get distribution weights.
     - Redistribute gathered descriptors back to positions:
         Y = G @ Cq, shape (B, hidden, N).

4) Output projection
     - Reshape Y to (B, hidden, H, W), then project back to channel dimension C
         using 1x1 conv + BatchNorm.
     - Final output shape is (B, C, H, W).

Why this design:
- Captures long-range/global dependencies with efficient matrix operations.
- Decouples context gathering and context distribution for clearer modeling.
- Maintains spatial resolution while enhancing feature interaction.

ZH:
A2 模块（Double Attention，双重注意力）通过“两阶段注意力聚合”先收集全局上下文，
再把上下文分配回各空间位置。可以理解为：
“先全局汇聚（gather），再空间分发（distribute）”。

整体结构：
1）输入投影
     - 对输入 X（形状为 (B, C, H, W)）分别做三个 1x1 卷积映射：
         A = ConvA(X), Bf = ConvB(X), Cq = ConvC(X)，形状均为 (B, hidden, H, W)。
     - 将空间维展平为 N = H*W，得到 (B, hidden, N)。

2）第一次注意力（特征汇聚）
     - 在 A 的空间维 N 上做 softmax，得到汇聚权重。
     - 通过批量矩阵乘法计算全局描述：
         G = Bf @ A^T，形状为 (B, hidden, hidden)。
     - 这一步把空间信息压缩为紧凑的全局特征表示。

3）第二次注意力（特征分发）
     - 在 Cq 的空间维 N 上做 softmax，得到分发权重。
     - 将全局描述重新分配到各空间位置：
         Y = G @ Cq，形状为 (B, hidden, N)。

4）输出投影
     - 将 Y 还原为 (B, hidden, H, W)，再通过 1x1 卷积 + BatchNorm
         映射回通道维 C。
     - 最终输出形状为 (B, C, H, W)。

设计动机：
- 用较高效的矩阵运算建模长程/全局依赖。
- 将“信息汇聚”和“信息分发”解耦，建模更清晰。
- 在不降低空间分辨率的前提下提升特征交互能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class A2(nn.Module):
    def __init__(self, channels, reduction=4, L = 32):
        super().__init__()
        
        assert channels > 0
        hidden = max(L, channels//reduction)
        self.channels = channels
        
        self.hidden = hidden
        
        self.ConvA = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ConvB = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ConvC = nn.Conv2d(channels, hidden, 1, bias=False)
        
        self.Proj = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # 将输入特征图通过三个不同的 1x1 卷积映射到隐藏维度，得到三个张量 A_tensor、B_feat 和 C_tensor，形状均为 (B, hidden, N)，其中 N=H*W 是空间维度展开后的长度。
        A_tensor = self.ConvA(x).view(B, self.hidden, N) # (B, hidden, N)
        B_feat = self.ConvB(x).view(B, self.hidden, N) # (B , hidden, N)
        C_tensor = self.ConvC(x).view(B, self.hidden, N) # (B, hidden, N)
        
        A_tensor = F.softmax(A_tensor, dim=2) # (B, hidden, N)
        # 第一次融合：将 B_feat 与 A_tensor 的转置进行批量矩阵乘法，得到 FeatureGathering 张量，形状为 (B, hidden, hidden)。这个操作相当于在空间维度上对 B_feat 进行加权求和，权重由 A_tensor 提供。
        FeatureGathering = torch.bmm(B_feat, A_tensor.permute(0, 2, 1)) # (B, hidden, hidden)
        
        V_attn = F.softmax(C_tensor, dim=2) # (B, hidden, N)
        # 第二次融合：将 FeatureGathering 与 V_attn 进行批量矩阵乘法，得到 x 张量，形状为 (B, hidden, N)。这个操作相当于在空间维度上对 FeatureGathering 进行加权求和，权重由 V_attn 提供。
        x = torch.bmm(FeatureGathering, V_attn) # (B, hidden, N)
        # 最后将 x 张量重新调整为 (B, hidden, H, W) 的形状，并通过一个 1x1 卷积和批归一化层进行投影，得到最终输出，形状为 (B, C, H, W)。
        x = x.view(B, self.hidden, H, W) # (B, hidden, H, W)
        # 将融合后的特征图通过一个 1x1 卷积和批归一化层进行投影，得到最终输出，形状为 (B, C, H, W)。
        out = self.Proj(x) # (B, C, H, W)
        
        return out
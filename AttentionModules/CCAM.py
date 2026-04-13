"""
EN:
CCAM (Convolutional Coordinate Attention Module) combines Channel Attention (CAM)
and Coordinate Attention (CA) to refine features from two complementary views:
- channel-wise importance (what features are important), and
- direction-aware spatial importance (where features are important along H/W axes).

High-level structure:
1) CAM branch (global channel reweighting)
     - Input X: (N, C, H, W).
     - Global average pooling -> (N, C, 1, 1).
     - 1x1 conv + sigmoid produce channel weights M_c: (N, C, 1, 1).
     - Apply channel gating: X_c = X * M_c.

2) CA branch (coordinate-aware spatial reweighting)
     - From X_c, perform directional pooling:
         H-branch: (N, C, H, 1), W-branch: (N, C, 1, W) -> permuted to (N, C, W, 1).
     - Concatenate along spatial axis to get Y: (N, C, H+W, 1).
     - Shared bottleneck transform (1x1 conv + BN + activation): C -> mip,
         where mip = max(8, C/reduction).
     - Split back into Y_h and Y_w, then project with two 1x1 convs (mip -> C)
         to obtain directional attention maps:
         A_h: (N, C, H, 1), A_w: (N, C, 1, W).

3) Fusion/output
     - Final output uses broadcast multiplication:
         Out = X * A_h * A_w.
     - This keeps original information flow while emphasizing informative channels
         and spatial coordinates.

Why this design:
- CAM improves global semantic channel selection.
- CA injects long-range positional cues with low overhead.
- Combining both often yields stronger feature discrimination than either alone.

ZH:
CCAM（Convolutional Coordinate Attention Module，卷积坐标注意力模块）
将通道注意力（CAM）与坐标注意力（CA）结合，从两个互补角度增强特征：
- 通道重要性（哪些特征更重要）；
- 方向感知的空间重要性（沿 H/W 方向哪些位置更重要）。

整体结构：
1）CAM 分支（全局通道重标定）
     - 输入 X 形状为 (N, C, H, W)。
     - 先做全局平均池化得到 (N, C, 1, 1)。
     - 经过 1x1 卷积与 sigmoid 得到通道权重 M_c，形状为 (N, C, 1, 1)。
     - 执行通道门控：X_c = X * M_c。

2）CA 分支（坐标感知空间重标定）
     - 在 X_c 上进行方向池化：
         高度分支得到 (N, C, H, 1)，宽度分支得到 (N, C, 1, W)，
         再转置为 (N, C, W, 1)。
     - 在空间维拼接得到 Y: (N, C, H+W, 1)。
     - 共享瓶颈变换（1x1 卷积 + BN + 激活）：C -> mip，
         其中 mip = max(8, C/reduction)。
     - 再拆分为 Y_h 与 Y_w，并通过两个 1x1 卷积（mip -> C）映射回通道数，
         得到方向注意力图：A_h: (N, C, H, 1), A_w: (N, C, 1, W)。

3）融合与输出
     - 通过广播逐元素相乘得到最终输出：
         Out = X * A_h * A_w。
     - 该形式在保留原始信息通路的同时，增强关键通道与关键位置响应。

设计动机：
- CAM 负责全局语义层面的通道筛选。
- CA 以较低开销引入长程方向位置信息。
- 二者结合通常比单独使用任一模块获得更强的特征判别能力。
"""

import torch.nn as nn
import torch

class FastSigmoid(nn.Module):
    def __init__(self):
        super(FastSigmoid, self).__init__()
        self.ReLU6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Fast Sigmoid approximation, using ReLU6 to avoid overflow
        # 快速 Sigmoid 近似，使用 ReLU6 避免溢出
        return self.ReLU6(x + 3) / 6

class FastSwish(nn.Module):
    def __init__(self):
        super(FastSwish, self).__init__()
        self.f_sigmoid = FastSigmoid()

    def forward(self, x):
        # Fast Swish approximation
        # 快速 Swish 近似
        return x * self.f_sigmoid(x)

class CCAM(nn.Module):
    """
    Convolutional Coordinate Attention Module (CCAM) for enhanced feature extraction.

    CCAM is a combination of Channel Attention Module (CAM) and Coordinate Attention (CA) for improved feature representation,
    particularly in convolutional neural networks. It enhances the model's ability to focus on important features by
    applying attention mechanisms in both channel and spatial dimensions.

    卷积坐标注意力模块(CCAM)用于增强特征提取。

    CCAM 是通道注意力模块(CAM)和坐标注意力(CA)的组合，用于改进特征表示，
    特别是在卷积神经网络中。通过在通道和空间维度上应用注意力机制，它增强了模型关注重要特征的能力。
    """

    def __init__(self, c1, reduction=16):
        """
        Initialize CCAM module.

        初始化 CCAM 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            reduction (int): Reduction ratio for channel attention. 通道注意力的缩减比例。
        """
        super(CCAM, self).__init__()
        # Initialize the channel attention.
        # 初始化通道注意力
        self.cam_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=1, stride=1, bias=True)
        self.ReLU = nn.Sigmoid()

        # Initialize the coordinate attention.
        # 初始化坐标注意力
        self.ca_avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.ca_avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=mip, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(mip)
        self.act = FastSwish()
        # Height-wise and width-wise convolutional layers
        # 高度和宽度卷积层
        self.conv_h = nn.Conv2d(in_channels=mip, out_channels=c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels=mip, out_channels=c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass through CCAM module.

        通过 CCAM 模块执行前向传播。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after applying CCAM. 应用 CCAM 后的输出张量。
        """
        temp = x

        # Channel Attention (CAM) forward pass
        # 通道注意力（CAM）前向传播
        cam_avg_out = self.ReLU(self.fc1(self.cam_avg_pool(x)))
        x = x * cam_avg_out

        # Coordinate Attention (CA) forward pass
        # 坐标注意力（CA）前向传播
        n, c, h, w = x.size()
        ca_avg_h_out = self.ca_avg_pool_h(x)
        ca_avg_w_out = self.ca_avg_pool_w(x).permute(0, 1, 3, 2)
        # Concatenate along the channel dimension
        # 在通道维度上连接
        y = torch.cat([ca_avg_h_out, ca_avg_w_out], dim=2)
        y = self.conv1(y)
        y = self.BN(y)
        y = self.act(y)
        # Split the output into height and width components
        # 将输出拆分为高度和宽度组件
        ca_h_out, ca_w_out = torch.split(y, [h, w], dim=2)
        ca_w_out = ca_w_out.permute(0, 1, 3, 2)
        # Apply the coordinate attention
        # 应用坐标注意力
        h_attention = self.conv_h(ca_h_out).sigmoid()
        w_attention = self.conv_w(ca_w_out).sigmoid()

        # Apply the attention to the input tensor
        # 将注意力应用于输入张量
        output = temp * h_attention * w_attention

        return output
"""
EN:
SimAM (Simple, Parameter-Free Attention Module) generates element-wise attention
without adding learnable convolution/MLP parameters.
It evaluates the importance of each neuron using a simple energy-based form
derived from local statistics, then applies sigmoid gating.

Core idea:
- For each channel, compare each spatial response with the channel mean.
- Larger normalized deviation from the mean implies higher saliency.
- Use a closed-form scoring function, avoiding extra parameterized layers.

High-level structure:
1) Compute per-channel mean
     - Input X: (B, C, H, W).
     - Mean over spatial dimensions:
         mu = mean(X, dim=(H, W)), shape (B, C, 1, 1).

2) Measure per-element deviation
     - Squared deviation:
         d = (X - mu)^2, shape (B, C, H, W).
     - Channel-wise variance estimate:
         var = mean(d, dim=(H, W)), shape (B, C, 1, 1).

3) Energy-inspired score and attention
     - Score in this implementation:
         score = d / (4 * (var + lambda)) + 0.5
     - lambda (e_lambda) is a small constant for numerical stability.
     - Apply sigmoid to obtain attention map A in (0, 1):
         A = sigmoid(score), shape (B, C, H, W).

4) Reweight feature map
     - Output:
         Out = X * A.

Why this design:
- Parameter-free attention with almost no extra model size.
- Keeps full spatial resolution and produces fine-grained element-wise weights.
- Easy to insert into existing CNN blocks with low overhead.

ZH:
SimAM（Simple, Parameter-Free Attention Module，简单无参数注意力模块）
在不引入额外可学习卷积或 MLP 参数的前提下，生成逐元素注意力权重。
它基于局部统计量构造能量函数形式的评分，再通过 sigmoid 完成门控。

核心思想：
- 对每个通道，比较每个空间位置与该通道均值的偏离程度；
- 归一化偏离越大，通常表示该响应越显著；
- 采用闭式评分表达式，不依赖额外参数层。

整体结构：
1）计算通道均值
     - 输入 X 形状为 (B, C, H, W)。
     - 在空间维求均值得到：
         mu = mean(X, dim=(H, W))，形状 (B, C, 1, 1)。

2）计算逐元素偏差
     - 偏差平方：
         d = (X - mu)^2，形状 (B, C, H, W)。
     - 通道方差估计：
         var = mean(d, dim=(H, W))，形状 (B, C, 1, 1)。

3）能量式评分与注意力
     - 本实现中的评分为：
         score = d / (4 * (var + lambda)) + 0.5。
     - 其中 lambda（e_lambda）为数值稳定项。
     - 经过 sigmoid 得到注意力图 A（取值在 0 到 1）：
         A = sigmoid(score)，形状 (B, C, H, W)。

4）特征重标定
     - 最终输出：
         Out = X * A。

设计动机：
- 无参数注意力，几乎不增加模型参数量。
- 保持完整空间分辨率，生成细粒度逐元素权重。
- 易于插入现有卷积网络模块，额外开销小。
"""

import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super().__init__()
        self.channels = channels
        self. e_lambda = e_lambda
        self. sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 通道均值
        mu = x.mean(dim=(2,3), keepdim=True) # (B,C,1,1)
        
        # 偏差平方
        d = (x - mu).pow(2) # (B,C,H,W)
        
        # 方差
        var = d.mean(dim=(2,3), keepdim=True) # (B,C,1,1)
        
        score = d / (4.0 * (var + self.e_lambda)) + 0.5 # 确保分数不过大不过小
        
        attention = self.sigmoid(score)
        return x * attention
    
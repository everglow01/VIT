"""
EN:
DANet (Dual Attention Network) applies two parallel self-attention modules —
one over the channel dimension and one over the spatial dimension — then fuses
their outputs to capture long-range contextual dependencies.

High-level structure:
1) Position Attention Module (PAM)
     - Input feature X: (N, C, H, W). Reshape to (N, C, HW).
     - Compute a spatial affinity matrix S: (N, HW, HW) via query-key dot product.
     - Apply softmax over the last dimension to get attention weights.
     - Aggregate values with the attention weights and reshape back to (N, C, H, W).
     - Scale by a learnable parameter gamma_p (init 0) and add residual: out_p = gamma_p * attn + X.

2) Channel Attention Module (CAM)
     - Reshape X to (N, C, HW).
     - Compute a channel affinity matrix E: (N, C, C) via key-query dot product.
     - Apply softmax to get channel attention weights.
     - Aggregate and reshape back to (N, C, H, W).
     - Scale by a learnable parameter gamma_c (init 0) and add residual: out_c = gamma_c * attn + X.

3) Fusion
     - Element-wise sum: out = out_p + out_c.

Why this design:
- PAM captures long-range spatial dependencies regardless of distance.
- CAM models inter-channel semantic correlations.
- Learnable gamma allows the network to gradually incorporate attention.

ZH:
DANet（双注意力网络）并行运行两个自注意力模块——一个作用于空间维度，一个作用于通道维度，
最后将两路输出融合，以捕获长程上下文依赖关系。

整体结构：
1）位置注意力模块（PAM）
     - 输入特征 X 形状为 (N, C, H, W)，展平为 (N, C, HW)。
     - 通过 query-key 点积计算空间亲和矩阵 S，形状为 (N, HW, HW)。
     - 对最后一维做 softmax 得到注意力权重。
     - 用注意力权重聚合 value，再 reshape 回 (N, C, H, W)。
     - 乘以可学习参数 gamma_p（初始化为 0）并加残差：out_p = gamma_p * attn + X。

2）通道注意力模块（CAM）
     - 将 X 展平为 (N, C, HW)。
     - 通过 key-query 点积计算通道亲和矩阵 E，形状为 (N, C, C)。
     - 对最后一维做 softmax 得到通道注意力权重。
     - 聚合后 reshape 回 (N, C, H, W)。
     - 乘以可学习参数 gamma_c（初始化为 0）并加残差：out_c = gamma_c * attn + X。

3）融合
     - 逐元素相加：out = out_p + out_c。

设计动机：
- PAM 不受距离限制地建模长程空间依赖。
- CAM 建模通道间的语义相关性。
- 可学习的 gamma 让网络逐步引入注意力，训练更稳定。
"""

import torch
import torch.nn as nn


class PAM(nn.Module):
    '''Position Attention Module — captures long-range spatial dependencies.'''
    '''位置注意力模块——捕获长程空间依赖关系。'''

    def __init__(self, C):
        super().__init__()
        hidden = max(1, C // 8)
        self.query = nn.Conv2d(C, hidden, 1, bias=False)
        self.key   = nn.Conv2d(C, hidden, 1, bias=False)
        self.value = nn.Conv2d(C, C,      1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        HW = H * W

        # query: (B, HW, hidden), key: (B, hidden, HW)
        q = self.query(x).view(B, -1, HW).permute(0, 2, 1)
        k = self.key(x).view(B, -1, HW)
        v = self.value(x).view(B, -1, HW)

        # spatial affinity matrix: (B, HW, HW)
        S = self.softmax(torch.bmm(q, k))

        # aggregate: (B, C, HW) -> (B, C, H, W)
        out = torch.bmm(v, S.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class CAM(nn.Module):
    '''Channel Attention Module — models inter-channel semantic correlations.'''
    '''通道注意力模块——建模通道间的语义相关性。'''

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        HW = H * W

        # reshape to (B, C, HW)
        feat = x.view(B, C, HW)

        # channel affinity matrix: (B, C, C)
        E = self.softmax(torch.bmm(feat, feat.permute(0, 2, 1)))

        # aggregate: (B, C, HW) -> (B, C, H, W)
        out = torch.bmm(E, feat).view(B, C, H, W)
        return self.gamma * out + x


class DANet(nn.Module):
    '''Dual Attention Network (DANet) fusing position and channel attention.'''
    '''双注意力网络（DANet），融合位置注意力与通道注意力。'''

    def __init__(self, C):
        super().__init__()
        self.pam = PAM(C)
        self.cam = CAM()

    def forward(self, x):
        return self.pam(x) + self.cam(x)

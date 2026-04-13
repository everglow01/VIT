"""
EN:
SCSA is a two-stage attention block that combines
1) spatially-aware multi-scale axis attention (SMSA), and
2) pooled channel self-attention (PCSA).
It first builds direction-aware spatial masks from multi-scale 1D depthwise
convolutions, then applies channel refinement on a pooled feature map.

High-level structure:
1) SMSA: multi-scale spatial-direction attention
     - Input X: (B, C, H, W), split channels into n groups (each Cg = C/n).
     - Height descriptor: average over width -> (B, C, H).
     - Width descriptor: average over height -> (B, C, W).
     - Split both descriptors into n channel chunks and process each chunk with
         depthwise Conv1d using different kernel sizes (multi-scale receptive fields).
     - Concatenate chunk outputs back to (B, C, H) and (B, C, W), normalize
         (GroupNorm) and apply sigmoid to get Ah and Aw.
     - Construct spatial mask by outer-product style broadcasting:
         Ms = Ah[..., None] * Aw[:, :, None, :], shape (B, C, H, W).
     - Spatially refined feature: X_s = X * Ms.

2) PCSA: pooled channel self-attention
     - Downsample X_s with adaptive average pooling to (B, C, p, p) for efficiency.
     - Normalize pooled map, then apply CA_SHSA (channel-attention style self-attention)
         on pooled spatial grid.
     - Global average pool and sigmoid to produce channel gate Mc: (B, C, 1, 1).
     - Final output: Out = X_s * Mc.

3) Output
     - Output shape remains identical to input: (B, C, H, W).

Why this design:
- Multi-scale Conv1d branches capture directional context at different ranges.
- Axis-factorized spatial mask is efficient and expressive.
- Pooled channel self-attention reduces cost while preserving global channel cues.
- Sequential spatial-then-channel refinement improves feature selectivity.

ZH:
SCSA 是一个两阶段注意力模块，结合了：
1）空间多尺度轴向注意力（SMSA），以及
2）池化后的通道自注意力（PCSA）。
它先通过多尺度一维深度卷积构造方向感知空间掩码，再在池化特征上做通道细化。

整体结构：
1）SMSA：多尺度空间方向注意力
     - 输入 X 形状为 (B, C, H, W)，将通道分成 n 组（每组 Cg = C/n）。
     - 高度描述：沿宽度平均，得到 (B, C, H)。
     - 宽度描述：沿高度平均，得到 (B, C, W)。
     - 两种描述都按通道切成 n 份，每份分别经过不同核大小的深度 Conv1d，
         以建模不同尺度的方向上下文。
     - 将各分支结果拼接回 (B, C, H) 与 (B, C, W)，经 GroupNorm 与 sigmoid
         得到 Ah、Aw。
     - 通过广播构造外积式空间掩码：
         Ms = Ah[..., None] * Aw[:, :, None, :]，形状为 (B, C, H, W)。
     - 空间增强特征：X_s = X * Ms。

2）PCSA：池化通道自注意力
     - 为降低开销，先将 X_s 自适应池化到 (B, C, p, p)。
     - 归一化后送入 CA_SHSA（通道自注意力风格模块）建模通道关系。
     - 再经全局平均池化与 sigmoid 得到通道门控 Mc，形状为 (B, C, 1, 1)。
     - 最终输出：Out = X_s * Mc。

3）输出
     - 输出形状与输入一致： (B, C, H, W)。

设计动机：
- 多尺度 Conv1d 分支可捕捉不同范围的方向信息。
- 轴向分解的空间掩码高效且表达力强。
- 在池化空间做通道自注意力可降低计算开销。
- 先空间后通道的串联细化有助于提升特征判别能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CA_SHSA(nn.Module):
    def __init__(self, channels, pool_hw):
        super().__init__()
        # 标准 1x1 卷积，允许跨通道信息混合（原 groups=channels 等价于逐通道缩放，无跨通道交互）
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        # 预计算缩放因子，避免 forward 中重复计算
        self.scale = math.sqrt(pool_hw * pool_hw)

    def forward(self, x):
        B, C, Hp, Wp = x.shape
        N = Hp * Wp

        q = self.q(x).view(B, C, N)  # (B, C, N)
        k = self.k(x).view(B, C, N)  # (B, C, N)
        v = self.v(x).view(B, C, N)  # (B, C, N)
        # 除以 sqrt(N) 缩放点积，防止数值过大导致 softmax 梯度消失
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, C, C)
        attn = F.softmax(attn, dim=-1)  # (B, C, C)
        out = torch.bmm(attn, v).view(B, C, Hp, Wp)  # (B, C, Hp, Wp)

        return out

def _safe_gn_groups(channels, prefer=4):
    if channels % prefer == 0:
        return prefer
    for g in range(prefer, 0, -1):
        if channels % g == 0:
            return g
    return 1

class MS_DWConv1d(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size=k, padding=k//2, groups=channels, bias=False)
        
    def forward(self, x) -> torch.Tensor:
        return self.dw(x)

class SCSA(nn.Module):
    def __init__(self, channels, n = 4, kernels=(3,5,7,9,), pool_hw=7):
        super().__init__()
        assert channels > 0
        assert n > 0
        assert channels % n == 0, "Channels must be divisible by n"
        
        self.C = channels
        self.n = n
        self.cg = channels // n
        if isinstance(kernels, (list, tuple)):
            assert len(kernels) == n, "Length of kernels must match n"
        else:
            raise TypeError("kernels must be a list or tuple of length n")
        
        # SMSA
        self.ms_h = nn.ModuleList([MS_DWConv1d(self.cg, int(k)) for k in kernels])
        self.ms_w = nn.ModuleList([MS_DWConv1d(self.cg, int(k)) for k in kernels])
        
        g = _safe_gn_groups(channels, prefer=4)
        
        self.gn_h = nn.GroupNorm(g, channels)
        self.gn_w = nn.GroupNorm(g, channels)
        self.sigmoid = nn.Sigmoid()
        
        # PCSA
        self.pool_hw = int(pool_hw)
        self.gn1 = nn.GroupNorm(1, channels)
        self.ca_shsa = CA_SHSA(channels, pool_hw)
        self.avgpool = nn.AdaptiveAvgPool2d(1) # (B, C, 1, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.C, f"Expected input with {self.C} channels, got {C}"
        x_h = x.mean(dim=3)
        x_w = x.mean(dim=2)
        # split
        x_h_chunks = torch.chunk(x_h, self.n, dim=1)  # n x (B, cg, H)
        x_w_chunks = torch.chunk(x_w, self.n, dim=1)  # n x (B, cg, W)
        
        h_out = [self.ms_h[i](x_h_chunks[i]) for i in range(self.n)]  # n x (B, cg, H)
        w_out = [self.ms_w[i](x_w_chunks[i]) for i in range(self.n)]  # n x (B, cg, W)
        
        # concat
        h_cat = torch.cat(h_out, dim=1)  # (B, C, H)
        w_cat = torch.cat(w_out, dim=1)  # (B, C, W)
        
        Ah = self.sigmoid(self.gn_h(h_cat))  # (B, C, H)
        Aw = self.sigmoid(self.gn_w(w_cat))  # (B, C, W)
        
        Ms = Ah.unsqueeze(-1) * Aw.unsqueeze(-2)  # (B, C, H, W)
        x_s = x * Ms # (B, C, H, W)
        
        p = self.pool_hw
        xp = F.adaptive_avg_pool2d(x_s, (p, p))  # (B, C, p, p)
        xp = self.gn1(xp)  # (B, C, p, p)
        xp = self.ca_shsa(xp)  # (B, C, p, p)
        Mc = torch.sigmoid(self.avgpool(xp))  # (B, C, 1, 1 )
        out = x_s * Mc  # (B, C, H, W)
        return out
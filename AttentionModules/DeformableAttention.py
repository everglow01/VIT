"""
EN:
DeformableAttention implements the single-scale deformable attention mechanism
introduced in Deformable DETR (Zhu et al., ICLR 2021).

Unlike standard self-attention that computes interactions over all HxW spatial
positions (O((HW)^2) complexity), deformable attention restricts each query to
attend to only a small fixed set of K learned sampling points.  The positions of
those sampling points are predicted dynamically from the query content, so the
module combines the spatial flexibility of attention with the efficiency of sparse
convolution-like operations.

High-level structure:
1) Flatten and project
     - Input X: (B, C, H, W).
     - Flatten spatial dimensions to obtain token sequence: x_flat (B, N, C),
       where N = H * W.
     - Project x_flat to values V via a linear layer; reshape V back to
       (B, C, H, W) to keep it as a dense feature map for bilinear sampling.

2) Reference-point grid
     - Build a normalised ([-1, 1]) meshgrid of H*W reference points, one per
       spatial location, matching the grid_sample coordinate convention.
     - Shape: (1, N, 1, 1, 2).

3) Offset and weight prediction
     - Offset branch: Linear(C -> num_heads * num_points * 2) applied to x_flat.
       Outputs per-query spatial offsets (dx, dy) for every head and every
       sampling point.  Offsets are scaled by 2/max(H,W) so they correspond to
       roughly ±1 pixel at initialisation.
       Shape after reshape: (B, N, num_heads, num_points, 2).
     - Attention-weight branch: Linear(C -> num_heads * num_points) applied to
       x_flat, followed by softmax over the num_points dimension.
       Shape after reshape: (B, N, num_heads, num_points).

4) Bilinear sampling
     - Sampling coordinates = reference points + offsets, clamped to [-1, 1].
     - Reshape value map to (B*num_heads, head_dim, H, W) and coordinates to
       (B*num_heads, N, num_points, 2).
     - F.grid_sample produces (B*num_heads, head_dim, N, num_points).

5) Weighted aggregation
     - Multiply sampled features by attention weights and sum over num_points.
     - Reshape result to (B, N, C) and apply output linear projection.

6) Residual + LayerNorm
     - out = LayerNorm(x_flat + dropout(projected_out)).
     - Reshape back to (B, C, H, W).

Why this design:
- Restricting attention to K points per query reduces complexity from O((HW)^2)
  to O(HW * K), making it practical for high-resolution feature maps.
- Learned offsets let the network adaptively focus on the most informative
  locations rather than being forced to attend uniformly or locally.
- Bilinear interpolation via grid_sample provides sub-pixel precision and
  differentiable gradient flow through the sampling positions.
- Zero-initialised offset weights ensure the module behaves like a regular
  dense attention at the start of training and gradually learns sparse patterns.

ZH:
DeformableAttention 实现了 Deformable DETR（Zhu et al., ICLR 2021）中提出的
单尺度可变形注意力机制。

标准自注意力需要对所有 HxW 个空间位置两两计算交互（复杂度 O((HW)^2)），而可变形
注意力让每个 query 只与 K 个动态预测的采样点做交互。采样点的位置由 query 的内容
动态决定，从而兼顾了注意力的空间灵活性与类稀疏卷积的计算高效性。

整体结构：
1）展平与投影
     - 输入 X：(B, C, H, W)。
     - 展平空间维度得到 token 序列 x_flat：(B, N, C)，N = H * W。
     - 用线性层将 x_flat 投影为值 V，再重塑回 (B, C, H, W) 以便双线性采样。

2）参考点网格
     - 在 H*W 个空间位置上构建归一化（[-1, 1]）的参考点网格，
       坐标约定与 grid_sample 一致。
     - 形状：(1, N, 1, 1, 2)。

3）偏移量与注意力权重预测
     - 偏移量分支：对 x_flat 做 Linear(C -> num_heads * num_points * 2)，
       输出每个 query 在各头各采样点上的空间偏移 (dx, dy)。
       偏移量按 2/max(H,W) 缩放，初始化时对应约 ±1 个像素的偏移范围。
       重塑后形状：(B, N, num_heads, num_points, 2)。
     - 注意力权重分支：对 x_flat 做 Linear(C -> num_heads * num_points)，
       再在 num_points 维做 softmax。
       重塑后形状：(B, N, num_heads, num_points)。

4）双线性采样
     - 采样坐标 = 参考点 + 偏移量，截断到 [-1, 1]。
     - 将值特征图重塑为 (B*num_heads, head_dim, H, W)，
       坐标重塑为 (B*num_heads, N, num_points, 2)。
     - F.grid_sample 输出 (B*num_heads, head_dim, N, num_points)。

5）加权聚合
     - 采样特征乘以注意力权重后在 num_points 维求和。
     - 重塑结果为 (B, N, C) 并经输出线性层投影。

6）残差连接 + LayerNorm
     - out = LayerNorm(x_flat + dropout(投影后输出))。
     - 重塑回 (B, C, H, W)。

设计动机：
- 每个 query 只关注 K 个采样点，将复杂度从 O((HW)^2) 降至 O(HW * K)，
  适用于高分辨率特征图。
- 可学习的偏移量使网络自适应地聚焦于最有信息量的位置，而非均匀或固定的局部区域。
- grid_sample 的双线性插值提供亚像素精度并支持对采样位置的梯度反传。
- 偏移量投影层权重初始化为零，确保训练初期模块行为接近稠密注意力，
  之后逐渐学习稀疏的注意力模式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttention(nn.Module):
    """Single-scale deformable attention from Deformable DETR.  Input/output: (B, C, H, W)."""
    """单尺度可变形注意力（来自 Deformable DETR），输入/输出均为 (B, C, H, W)。"""

    def __init__(self, d_model, num_heads=8, num_points=4, dropout=0.0):
        """
        Args:
            d_model    : number of input/output channels C.
                         通道数 C，即 d_model。
            num_heads  : number of attention heads; d_model must be divisible by num_heads.
                         注意力头数；d_model 必须能被 num_heads 整除。
            num_points : number of sampling points per query per head (K).
                         每个 query 每个头的采样点数 K。
            dropout    : dropout probability applied after output projection.
                         输出投影后的 dropout 概率。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = d_model // num_heads

        # 值投影：将输入特征线性映射到值空间
        self.value_proj = nn.Linear(d_model, d_model)

        # 偏移量预测：每个 query 输出 num_heads × num_points 个 (dx, dy) 偏移
        self.offset_proj = nn.Linear(d_model, num_heads * num_points * 2)

        # 注意力权重预测：每个 query 输出 num_heads × num_points 个标量权重
        self.attn_weight_proj = nn.Linear(d_model, num_heads * num_points)

        # 输出投影：聚合后的特征映射回 d_model 维
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        # 偏移量投影初始化为零：训练初期采样点与参考点重合，类似均匀采样
        # 随着训练推进，网络自动学习偏移稀疏注意力模式
        nn.init.constant_(self.offset_proj.weight, 0.0)
        nn.init.constant_(self.offset_proj.bias, 0.0)
        # 注意力权重投影初始化为零：softmax(zeros) = uniform(1/num_points)
        # 确保训练初期各采样点权重相等，避免随机初始化导致的权重分布不均匀
        nn.init.constant_(self.attn_weight_proj.weight, 0.0)
        nn.init.constant_(self.attn_weight_proj.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x : (B, C, H, W) — input feature map.
                输入特征图。
        Returns:
            out : (B, C, H, W) — attended feature map, same shape as input.
                  与输入同形的注意力输出特征图。
        """
        B, C, H, W = x.shape
        N = H * W  # 空间位置总数

        # ── 1. 展平空间维度，得到 token 序列 ───────────────────────────────
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # ── 2. 值投影，重塑为特征图供 grid_sample 使用 ───────────────────────
        v = self.value_proj(x_flat)                          # (B, N, C)
        v = v.transpose(1, 2).view(B, C, H, W)              # (B, C, H, W)

        # ── 3. 参考点网格（[-1, 1] 归一化坐标，与 grid_sample 约定一致） ──────
        # grid_x 对应列（W 方向），grid_y 对应行（H 方向）
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij',
        )
        # ref_points: (1, N, 1, 1, 2)，广播到所有 batch/head/point
        ref_points = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        ref_points = ref_points.reshape(1, N, 1, 1, 2)      # (1, N, 1, 1, 2)

        # ── 4. 偏移量预测 ───────────────────────────────────────────────────
        offsets = self.offset_proj(x_flat)                                    # (B, N, num_heads*num_points*2)
        offsets = offsets.view(B, N, self.num_heads, self.num_points, 2)      # (B, N, nh, np, 2)
        # 缩放因子：将偏移量映射到归一化坐标系下约 ±1 像素的范围
        offsets = offsets * (2.0 / max(H, W))

        # ── 5. 注意力权重预测 ────────────────────────────────────────────────
        attn_w = self.attn_weight_proj(x_flat)                                # (B, N, num_heads*num_points)
        attn_w = attn_w.view(B, N, self.num_heads, self.num_points)           # (B, N, nh, np)
        attn_w = F.softmax(attn_w, dim=-1)                                    # 在 num_points 维归一化

        # ── 6. 计算采样坐标，夹断到合法范围 ────────────────────────────────────
        # ref_points 广播到 (B, N, num_heads, num_points, 2)
        coords = ref_points + offsets          # (B, N, nh, np, 2)
        coords = coords.clamp(-1.0, 1.0)

        # ── 7. 批量双线性采样（合并 batch 与 heads 维以向量化） ──────────────────
        # 值特征图拆分为各头：(B, nh, head_dim, H, W) → (B*nh, head_dim, H, W)
        v_heads = v.view(B, self.num_heads, self.head_dim, H, W)
        v_heads = v_heads.flatten(0, 1)                                        # (B*nh, hd, H, W)

        # 采样坐标重排：(B, N, nh, np, 2) → (B, nh, N, np, 2) → (B*nh, N, np, 2)
        coords_bnh = coords.permute(0, 2, 1, 3, 4).flatten(0, 1)              # (B*nh, N, np, 2)

        # grid_sample: input (B*nh, hd, H, W), grid (B*nh, N, np, 2)
        # → 输出 (B*nh, hd, N, np)
        sampled = F.grid_sample(
            v_heads, coords_bnh,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )                                                                       # (B*nh, hd, N, np)

        # ── 8. 注意力加权求和 ────────────────────────────────────────────────
        # attn_w: (B, N, nh, np) → (B, nh, N, np) → (B*nh, 1, N, np)
        w = attn_w.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)             # (B*nh, 1, N, np)
        # 加权后在 num_points 维求和：(B*nh, hd, N)
        out = (sampled * w).sum(dim=-1)                                        # (B*nh, hd, N)

        # ── 9. 重塑并输出投影 ────────────────────────────────────────────────
        # (B*nh, hd, N) → (B, nh, hd, N) → (B, N, nh*hd) = (B, N, C)
        out = out.view(B, self.num_heads, self.head_dim, N)
        out = out.permute(0, 3, 1, 2).flatten(2)                               # (B, N, C)
        out = self.out_proj(out)                                                # (B, N, C)

        # ── 10. 残差连接 + LayerNorm ──────────────────────────────────────────
        out = self.norm(x_flat + self.dropout(out))                            # (B, N, C)

        # 重塑回特征图形式
        out = out.transpose(1, 2).view(B, C, H, W)                            # (B, C, H, W)
        return out

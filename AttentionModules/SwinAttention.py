"""
EN:
SwinAttention implements the Window-based Multi-head Self-Attention (W-MSA) and
Shifted Window Multi-head Self-Attention (SW-MSA) from Swin Transformer.
Features are partitioned into non-overlapping local windows, and self-attention
is computed within each window. In alternating layers, windows are shifted by
(window_size // 2) to enable cross-window connections.

High-level structure:
1) Window Partition
     - Input feature X: (N, H, W, C).
     - Partition into non-overlapping windows of size (M, M):
         shape becomes (num_windows * N, M, M, C).

2) Window Attention
     - For each window, compute Q, K, V via linear projections.
     - Add learnable relative position bias B to attention logits.
     - Attention: softmax((QK^T / sqrt(d)) + B) * V.
     - Output projected back to C dimensions.

3) Window Reverse
     - Merge windows back to (N, H, W, C).

4) Cyclic Shift (SW-MSA only)
     - Before partitioning, shift features by (-shift_size, -shift_size).
     - Apply attention mask to prevent cross-region attention.
     - Reverse shift after attention.

Why this design:
- Local window attention reduces complexity from O(N^2) to O(N * M^2).
- Shifted windows enable cross-window information exchange without extra cost.
- Relative position bias captures spatial relationships within each window.

ZH:
SwinAttention 实现了 Swin Transformer 中的窗口多头自注意力（W-MSA）和
移位窗口多头自注意力（SW-MSA）。特征被划分为不重叠的局部窗口，在每个窗口内
独立计算自注意力。在交替层中，窗口偏移 (window_size // 2) 以实现跨窗口连接。

整体结构：
1）窗口划分
     - 输入特征 X 形状为 (N, H, W, C)。
     - 划分为大小为 (M, M) 的不重叠窗口：
         形状变为 (num_windows * N, M, M, C)。

2）窗口注意力
     - 对每个窗口，通过线性投影计算 Q、K、V。
     - 在注意力 logits 上加入可学习的相对位置偏置 B。
     - 注意力：softmax((QK^T / sqrt(d)) + B) * V。
     - 输出通过线性层投影回 C 维。

3）窗口还原
     - 将窗口合并回 (N, H, W, C)。

4）循环移位（仅 SW-MSA）
     - 划分前将特征循环移位 (-shift_size, -shift_size)。
     - 应用注意力掩码防止跨区域注意力。
     - 注意力计算后反向移位还原。

设计动机：
- 局部窗口注意力将复杂度从 O(N^2) 降至 O(N * M^2)。
- 移位窗口在不增加额外开销的情况下实现跨窗口信息交互。
- 相对位置偏置捕获窗口内的空间关系。
"""

import torch
import torch.nn as nn


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.
    将特征图划分为不重叠的窗口。

    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to feature map.
    将窗口还原为特征图。

    Args:
        windows: (num_windows * B, window_size, window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    '''Window-based multi-head self-attention with relative position bias.'''
    '''基于窗口的多头自注意力，含相对位置偏置。'''

    def __init__(self, C, window_size=7, num_heads=8):
        super().__init__()
        self.num_heads = num_heads # 每个头的维度为 C // num_heads
        self.scale = (C // num_heads) ** -0.5 # 缩放因子，防止点积过大导致梯度消失
        self.window_size = window_size

        # relative position bias table: (2M-1) * (2M-1), num_heads
        M = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * M - 1) * (2 * M - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # precompute relative position index
        coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing='ij'))  # (2, M, M)
        coords_flat = coords.flatten(1)  # (2, M^2)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, M^2, M^2)
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += M - 1
        relative[:, :, 1] += M - 1
        relative[:, :, 0] *= 2 * M - 1
        self.register_buffer('relative_position_index', relative.sum(-1))  # (M^2, M^2)

        self.qkv = nn.Linear(C, C * 3, bias=True)
        self.proj = nn.Linear(C, C)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] # type: ignore
        bias = bias.view(self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        return self.proj(self.softmax(attn) @ v).reshape(B_, N, C)


class SwinAttention(nn.Module):
    '''Swin Transformer attention block with optional cyclic shift (SW-MSA).'''
    '''Swin Transformer 注意力块，支持可选的循环移位（SW-MSA）。'''

    def __init__(self, C, H, W, window_size=7, num_heads=8, shift=False):
        super().__init__()
        assert H % window_size == 0 and W % window_size == 0, \
            "H and W must be divisible by window_size"
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm = nn.LayerNorm(C)
        self.attn = WindowAttention(C, window_size, num_heads)

        # precompute attention mask for shifted windows
        if self.shift_size > 0:
            M, s = window_size, self.shift_size
            img_mask = torch.zeros(1, H, W, 1)
            # for h_slice, w_slice in [
            #     (slice(0, -M), slice(0, -M)),
            #     (slice(0, -M), slice(-M, -s)),
            #     (slice(0, -M), slice(-s, None)),
            #     (slice(-M, -s), slice(0, -M)),
            #     (slice(-M, -s), slice(-M, -s)),
            #     (slice(-M, -s), slice(-s, None)),
            #     (slice(-s, None), slice(0, -M)),
            #     (slice(-s, None), slice(-M, -s)),
            #     (slice(-s, None), slice(-s, None)),
            # ]:
            #     img_mask[:, h_slice, w_slice, :] = [0,1,2,3,4,5,6,7,8].pop(0)

            # # rebuild with correct region labels
            # img_mask = torch.zeros(1, H, W, 1)
            cnt = 0
            for h in [slice(0, -M), slice(-M, -s), slice(-s, None)]:
                for w in [slice(0, -M), slice(-M, -s), slice(-s, None)]:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        # x: (B, C, H, W) — convert to (B, H, W, C) for window ops
        _, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        shortcut = x
        x = self.norm(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition -> attention -> reverse
        windows = window_partition(x, self.window_size)                          # (nW*B, M, M, C)
        windows = windows.view(-1, self.window_size * self.window_size, C)       # (nW*B, M^2, C)
        attn_out = self.attn(windows, mask=self.attn_mask)                       # (nW*B, M^2, C)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, H, W)                     # (B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = (shortcut + x).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

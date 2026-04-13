"""
EN:
DETRAttention implements the Multi-head Attention used in DETR (Detection Transformer).
DETR applies standard Transformer encoder-decoder attention to object detection:
the encoder uses self-attention over image features, and the decoder uses cross-attention
between learned object queries and encoder memory.

This module provides the core attention block used in both encoder and decoder:
Multi-head Attention with optional cross-attention (query != key/value source).

High-level structure:
1) Linear projections
     - Project queries Q, keys K, values V via independent linear layers.
     - Split into num_heads heads: each head has dimension d_head = d_model // num_heads.

2) Scaled dot-product attention (per head)
     - Attention scores: A = softmax(QK^T / sqrt(d_head)).
     - Optional attention mask added before softmax (for padding).
     - Output: A * V.

3) Output projection
     - Concatenate all heads and project back to d_model via a linear layer.

4) Residual + LayerNorm
     - out = LayerNorm(x + dropout(attn_out)).

Usage:
     - Self-attention (encoder): query = key = value = src features.
     - Cross-attention (decoder): query = object queries, key = value = encoder memory.

Why this design:
- Object queries are learned positional embeddings that decode into individual objects.
- Cross-attention lets each query attend to relevant image regions globally.
- Standard Transformer attention with no inductive bias enables flexible detection.

ZH:
DETRAttention 实现了 DETR（Detection Transformer）中使用的多头注意力机制。
DETR 将标准 Transformer 编码器-解码器注意力应用于目标检测：
编码器对图像特征做自注意力，解码器在可学习的目标查询与编码器记忆之间做交叉注意力。

本模块提供编码器和解码器中共用的核心注意力块：
支持自注意力与交叉注意力（query 与 key/value 来源不同）的多头注意力。

整体结构：
1）线性投影
     - 通过独立线性层分别投影 Q、K、V。
     - 拆分为 num_heads 个头，每头维度 d_head = d_model // num_heads。

2）缩放点积注意力（每头独立计算）
     - 注意力分数：A = softmax(QK^T / sqrt(d_head))。
     - 可选在 softmax 前加注意力掩码（用于 padding）。
     - 输出：A * V。

3）输出投影
     - 拼接所有头的输出，经线性层投影回 d_model。

4）残差连接 + LayerNorm
     - out = LayerNorm(x + dropout(attn_out))。

使用方式：
     - 自注意力（编码器）：query = key = value = 图像特征。
     - 交叉注意力（解码器）：query = 目标查询，key = value = 编码器记忆。

设计动机：
- 目标查询是可学习的位置嵌入，每个查询解码为一个独立目标。
- 交叉注意力让每个查询在全局范围内关注相关图像区域。
- 标准 Transformer 注意力无归纳偏置，具有灵活的检测能力。
"""

import torch.nn as nn
import torch.nn.functional as F


class DETRAttention(nn.Module):
    '''Multi-head attention block used in DETR encoder/decoder.'''
    '''DETR 编码器/解码器中使用的多头注意力块。'''

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key=None, value=None, attn_mask=None):
        """
        Args:
            query: (B, Nq, d_model) — object queries (decoder) or src features (encoder)
            key:   (B, Nk, d_model) — encoder memory for cross-attn; defaults to query
            value: (B, Nk, d_model) — encoder memory for cross-attn; defaults to query
            attn_mask: (B, Nq, Nk) or (Nq, Nk), added before softmax (e.g. padding mask)
        Returns:
            out: (B, Nq, d_model)
        """
        # self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key

        B, Nq, _ = query.shape
        Nk = key.shape[1]
        h = self.num_heads

        # project and reshape to (B, h, N, d_head)
        q = self.q_proj(query).view(B, Nq, h, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, Nk, h, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, Nk, h, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, Nq, Nk)

        if attn_mask is not None:
            # broadcast mask across heads
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # aggregate and project
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, -1)  # (B, Nq, d_model)
        out = self.out_proj(out)

        # residual + norm
        return self.norm(query + self.dropout(out))

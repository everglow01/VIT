"""
EN:
ACmix combines local self-attention and dynamic convolution in a unified block.
Both branches share Q/K/V projections, then produce two complementary outputs:
- attention branch: content-adaptive weighted aggregation in local neighborhoods
- convolution branch: kernel-style aggregation with learned dynamic weights
The final feature is a learnable weighted fusion of the two branches.

High-level structure:
1) Shared projection
    - Input X: (B, C, H, W), where C is split into multiple heads.
    - 1x1 conv generates Q, K, V simultaneously:
      qkv = Conv1x1(X) -> (B, 3C, H, W), then split into q, k, v.

2) Attention branch (local multi-head attention)
    - Reshape q to per-head form.
    - Use unfold on k and v to extract kxk local patches for each location.
    - Compute attention logits by dot product between q and local k patches.
    - Softmax over neighborhood dimension (k^2), then aggregate v patches.
    - Output attn_out has shape (B, C, H, W).

3) Convolution branch (dynamic kernel aggregation)
    - Concatenate q, k, v and map them to per-head dynamic kernel weights.
    - For each location, apply weighted aggregation over local v patches,
      similar to dynamic depthwise convolution behavior.
    - Output conv_out has shape (B, C, H, W).

4) Fusion and projection
    - Learnable scalars alpha and beta balance two branches:
      out = alpha * attn_out + beta * conv_out.
    - Optional 1x1 projection refines/mixes channels.

Why this design:
- Attention branch improves content-aware dependency modeling.
- Convolution branch preserves strong locality and inductive bias.
- Shared projections reduce redundancy and improve efficiency.
- Learnable fusion adapts branch preference during training.

ZH:
ACmix 在同一个模块中融合了“局部自注意力”和“动态卷积”两条分支。
两条分支共享 Q/K/V 投影，再分别完成特征聚合：
- 注意力分支：基于内容相似度的局部自适应加权；
- 卷积分支：基于动态核权重的局部卷积式聚合；
最终输出由可学习系数对两分支结果加权融合得到。

整体结构：
1）共享投影
    - 输入 X 形状为 (B, C, H, W)，通道按多头方式划分。
    - 通过 1x1 卷积一次性生成 Q/K/V：
      qkv = Conv1x1(X) -> (B, 3C, H, W)，再拆分为 q、k、v。

2）注意力分支（局部多头注意力）
    - 将 q 按头重排。
    - 对 k、v 用 unfold 提取每个位置的 kxk 邻域 patch。
    - 用 q 与局部 k 做点积得到注意力 logits。
    - 在邻域维 k^2 上做 softmax，再对 v patch 加权求和。
    - 得到 attn_out，形状为 (B, C, H, W)。

3）卷积分支（动态核聚合）
    - 拼接 q、k、v 后映射出每头、每位置的动态卷积权重。
    - 对局部 v patch 做加权求和，行为类似动态深度卷积。
    - 得到 conv_out，形状为 (B, C, H, W)。

4）融合与投影
    - 用可学习系数 alpha、beta 融合两分支：
      out = alpha * attn_out + beta * conv_out。
    - 可选 1x1 投影进一步做通道混合与细化。

设计动机：
- 注意力分支强化内容自适应依赖建模能力。
- 卷积分支保留局部先验与稳定的卷积归纳偏置。
- 共享投影减少冗余计算，提高效率。
- 可学习融合使模型在训练中自适应平衡两类机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ACmix(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4, Proj=True):
        super().__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.pad = kernel_size // 2
        self.Proj = Proj
        
        # 共享投影产生 Q, K, V 1x1 卷积
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        self.kernel_proj = nn.Conv2d(3 * self.channels, self.num_heads * self.k2, kernel_size=1, groups=num_heads, bias=True)
        self.alpha = nn.Parameter(torch.zeros(0.5)) # 融合权重，初始值为0.5，训练过程中学习调整
        self.beta = nn.Parameter(torch.zeros(0.5))
        
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        qkv = self.qkv(x) # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1) # each (B, C, H, W)
        
        q_flat = q.view(B, self.num_heads, self.head_dim, L) # (B, num_heads, head_dim, L)
        # 把k v 展开成k*k领域patch
        k_patch= F.unfold(k, kernel_size=self.kernel_size, padding=self.pad).view(B, self.num_heads, self.head_dim, self.k2, L) # (B, num_heads, head_dim, k2, L)
        v_patch = F.unfold(v, kernel_size=self.kernel_size, padding=self.pad).view(B, self.num_heads, self.head_dim, self.k2, L) # (B, num_heads, head_dim, k2, L)
        # 计算注意力权重
        logits = (q_flat.unsqueeze(3)*k_patch).sum(dim=2) # (B, num_heads, k2, L)
        attn = F.softmax(logits, dim=2) # (B, num_heads, k2, L)
        attn_out = (attn.unsqueeze(2)*v_patch).sum(dim=3) # (B, num_heads, head_dim, L)
        attn_out = attn_out.view(B, self.channels, H, W) # (B, C, H, W)
        
        # 卷积分支
        kernel = self.kernel_proj(torch.cat([q, k, v], dim=1)).view(B, self.num_heads, self.k2, L) # (B, num_heads, k2, L)
        
        conv_out = (kernel.unsqueeze(2)*v_patch).sum(dim=3).view(B, self.channels, H, W) # (B, C, H, W)  
        out = self.alpha * attn_out + self.beta * conv_out
        if self.Proj is True:
            out = self.proj(out)
        return out
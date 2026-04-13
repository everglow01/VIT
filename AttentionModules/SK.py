"""
EN:
SK (Selective Kernel) Attention adaptively selects receptive fields by combining
multiple convolution branches with different kernel sizes.
Instead of using a fixed kernel scale, SK dynamically assigns channel-wise weights
to each branch according to the input content.

High-level structure:
1) Multi-branch feature extraction
    - Input X: (B, C, H, W).
    - Apply M parallel conv branches with different kernel sizes (e.g., 3x3, 5x5),
      each producing feature maps of shape (B, C, H, W).
    - Stack branch outputs to get F: (B, M, C, H, W).

2) Global descriptor generation
    - Fuse branches by summation over M: U = sum(F_i), shape (B, C, H, W).
    - Global average pooling: S = GAP(U), shape (B, C, 1, 1).
    - Bottleneck transform (1x1 conv + BN + activation):
      Z = phi(S), shape (B, hidden, 1, 1), where hidden = max(L, C/reduction).

3) Branch-wise attention generation
    - For each branch i, map Z to logits_i with a separate 1x1 conv:
      logits_i: (B, C, 1, 1).
    - Stack logits to (B, M, C, 1, 1), then apply softmax along branch dimension M.
    - This yields attention weights A that satisfy branch competition per channel.

4) Adaptive fusion
    - Weighted sum across branches:
      Out = sum(A_i * F_i), output shape (B, C, H, W).

Why this design:
- Dynamically adjusts receptive field size for different visual patterns.
- Captures multi-scale context with lightweight channel-wise gating.
- Often improves robustness to object scale variation.

ZH:
SK（Selective Kernel）注意力通过多分支不同卷积核并行建模，
实现“自适应感受野选择”。与固定卷积核不同，SK 会根据输入内容
为每个分支分配动态权重，从而按通道选择更合适的尺度信息。

整体结构：
1）多分支特征提取
    - 输入 X 形状为 (B, C, H, W)。
    - 使用 M 个并行卷积分支（如 3x3、5x5），每个分支输出 (B, C, H, W)。
    - 将分支结果堆叠为 F，形状为 (B, M, C, H, W)。

2）全局描述生成
    - 在分支维求和融合：U = sum(F_i)，形状 (B, C, H, W)。
    - 全局平均池化：S = GAP(U)，形状 (B, C, 1, 1)。
    - 经过瓶颈变换（1x1 卷积 + BN + 激活）得到
      Z = phi(S)，形状 (B, hidden, 1, 1)，其中 hidden = max(L, C/reduction)。

3）分支注意力生成
    - 每个分支使用独立 1x1 卷积将 Z 映射为 logits_i：
      形状 (B, C, 1, 1)。
    - 堆叠后得到 (B, M, C, 1, 1)，并在分支维 M 上做 softmax。
    - 得到的注意力权重在每个通道上形成分支竞争关系。

4）自适应融合
    - 对各分支特征做加权求和：
      Out = sum(A_i * F_i)，输出形状为 (B, C, H, W)。

设计动机：
- 能按输入内容动态调整有效感受野大小。
- 通过轻量门控实现多尺度信息自适应融合。
- 对目标尺度变化通常更鲁棒。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class SK(nn.Module):
    def __init__(self, channels, kernals=(3,5), reduction=16, L = 32, groups=1,act="silu"):
        super().__init__()
        self.channels = channels
        self.kernals = kernals
        self.M = len(kernals)
        self.hidden = max(L, channels//reduction)
        
        '''SK Attention: Selective Kernel Attention for Adaptive Receptive Fields.'''
        '''可以添加不同尺寸的卷积核来捕捉不同尺度的特征，并通过注意力机制自适应地融合这些特征。'''
        ''' 特殊情况也可以加入全局平均池化作为一个分支，捕捉全局上下文信息。'''
        # self.Conv1x1 = nn.Conv2d(channels, channels, 1, bias=False)
        # self.Conv3x3 = nn.Conv2d(channels, channels, 3,  bias=False)
        # self.Conv5x5 = nn.Conv2d(channels, channels, 5,  bias=False)
        
        # 激活函数
        if act.lower() =="silu":
            self.act = nn.SiLU(inplace=True)
        if act.lower() =="relu":
            self.act = nn.ReLU(inplace=True)
        if act.lower() =="hardswish":
            self.act = nn.Hardswish(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True) # 默认使用SiLU激活函数
            Warning("do not support this activation, use SiLU as default")
        
        # 初始化分支列表
        self.branches = nn.ModuleList()
        for k in kernals:
            p = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, k, padding=p, bias=False, groups=groups),
                    nn.BatchNorm2d(channels),
                    self.act
                )
            )
            
            # fuse
            self.fuse = nn.AdaptiveAvgPool2d(1) # (B,C,H,W) -> (B,C,1,1)      
            # generate attention
            self.fc = nn.Sequential(
                nn.Conv2d(channels, self.hidden, 1, bias=False),
                nn.BatchNorm2d(self.hidden),
                self.act
            )
        # 每个分支对应一个全连接层用于生成注意力权重 然后softmax
        self.fc_branhes = nn.ModuleList([nn.Conv2d(self.hidden, channels, 1, bias=False) for _ in range(self.M)])
        
        
    def forward(self, x):
        feat = [br(x) for br in self.branches] # 每个分支的输出列表
        feats_stack = torch.stack(feat, dim=1) # (B,M,C,H,W)
        
        U = feats_stack.sum(dim=1) # (B,C,H,W) 将所有分支的特征图相加得到融合特征图
        S = self.fuse(U) # (B,C,1,1) 对融合特征图进行全局平均池化得到全局描述
        Z = self.fc(S) # (B,hidden,1,1) 通过全连接层生成隐藏维度的特征表示
        
        logits = torch.stack([fc(Z) for fc in self.fc_branhes], dim=1) # (B,M,C,1,1) 每个分支对应一个全连接层生成注意力权重
        attention = F.softmax(logits, dim=1) # (B,M,C,1,1) 对注意力权重进行softmax归一化
        
        out = (feats_stack * attention).sum(dim=1) # (B,C,H,W) 将每个分支的特征图乘以对应的注意力权重后相加得到最终输出
        return out
"""
EN:
CBAM (Convolutional Block Attention Module) applies attention in two sequential
stages: channel attention first, then spatial attention. The module refines
features by answering two complementary questions:
- "What" is important?  (channel attention)
- "Where" is important? (spatial attention)

High-level structure:
1) Channel Attention
     - Input feature X: (N, C, H, W).
     - Perform both global average pooling and global max pooling to obtain two
         descriptors: (N, C, 1, 1).
     - Feed both descriptors into a shared bottleneck MLP-like transform
         (implemented by 1x1 conv layers): C -> C/r -> C.
     - Sum the two outputs and apply sigmoid to get channel map M_c: (N, C, 1, 1).
     - Reweight features: X_c = X * M_c (broadcast on H and W).

2) Spatial Attention
     - On X_c, compute channel-wise average and max projections to produce
         two maps: (N, 1, H, W) and (N, 1, H, W).
     - Concatenate to (N, 2, H, W), then apply a spatial convolution (typically 7x7)
         and sigmoid to obtain spatial map M_s: (N, 1, H, W).
     - Reweight features: X_s = X_c * M_s (broadcast on C).

Why this design:
- Pooling with avg+max captures complementary statistics.
- Sequential channel/spatial refinement is simple yet effective.
- Adds limited overhead while improving feature discrimination.

ZH:
CBAM（卷积块注意力模块）按顺序执行两步注意力：先通道注意力，再空间注意力。
它通过两个互补问题来增强特征：
- “看什么重要”（通道注意力）
- “在哪里重要”（空间注意力）

整体结构：
1）通道注意力
     - 输入特征 X 形状为 (N, C, H, W)。
     - 分别进行全局平均池化与全局最大池化，得到两个描述向量 (N, C, 1, 1)。
     - 两个描述向量通过共享瓶颈 MLP（用 1x1 卷积实现）：C -> C/r -> C。
     - 两路结果相加并经 sigmoid，得到通道注意力图 M_c，形状为 (N, C, 1, 1)。
     - 对输入进行通道重标定：X_c = X * M_c（在 H、W 维广播）。

2）空间注意力
     - 在 X_c 上沿通道维做平均与最大投影，得到两张图：
         (N, 1, H, W) 和 (N, 1, H, W)。
     - 拼接为 (N, 2, H, W)，经过空间卷积（常用 7x7）和 sigmoid，得到
         空间注意力图 M_s，形状为 (N, 1, H, W)。
     - 再进行空间重标定：X_s = X_c * M_s（在 C 维广播）。

设计动机：
- 平均池化与最大池化提供互补统计信息。
- 通道与空间顺序建模，结构直观且效果稳定。
- 额外开销较小，能显著提升特征表达能力。
"""

import torch
import torch.nn as nn

class SBAM(nn.Module):
    '''Convolutional Block Attention Module (CBAM) for enhanced feature representation.'''
    '''CBAM:增强特征表示的卷积块注意力模块。'''
    '''CBAM通过在通道和空间维度上应用注意力机制，增强了模型关注重要特征的能力。'''
    
    
    def __init__(self,C, reduction = 16, kernal_size = 7):
        
        
        '''1.Chanel Attention'''
        super().__init__()
        self.x1 = nn.AdaptiveMaxPool2d(1)
        self.x2 = nn.AdaptiveAvgPool2d(1)
        assert C > 0
        hidden = max(1, C//reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(C, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, C, 1, bias=False)
        )
        
        
        '''Spatial Attention'''
        assert kernal_size % 2 == 1, "kernal_size必须为奇数"
        pad = kernal_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernal_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
            
        B,C,H,W = x.shape
        '''1.Chanel Attention'''
        max = self.x1(x)
        avg = self.x2(x)
            
        Mc = self.sigmoid(self.mlp(avg) + self.mlp(max))
        x_c = x * Mc #广播到同形状
            
        '''2.spatial Attention'''
            
        avg_c = x_c.mean(dim=1, keepdim=True) # (B,1,H,W)
        max_c= x_c.max(dim=1, keepdim=True)[0] # (B,1,H,W) 
            
        # concat
        s = torch.cat([avg_c, max_c], dim=1) # (B,2,H,W)
        Sp = self.sigmoid(self.spatial_conv(s)) # (B,1,H,W)
        x_s = x * Sp #广播到同形状
        return x_s
"""
EN:
SE (Squeeze-and-Excitation) is a channel attention mechanism that adaptively
recalibrates channel responses with three key steps: Squeeze, Excitation, and Scale.
It improves representational power by explicitly modeling channel inter-dependencies.

High-level structure:
1) Squeeze
    - Input X: (N, C, H, W).
    - Global average pooling compresses spatial information into channel descriptors:
      Z: (N, C, 1, 1).

2) Excitation
    - Use a bottleneck two-layer transform (commonly viewed as MLP, implemented
      here by 1x1 convolutions): C -> C/r -> C.
    - Nonlinearity (e.g., ReLU) between the two layers.
    - Sigmoid produces channel weights S in [0, 1] with shape (N, C, 1, 1).

3) Scale
    - Reweight original feature channels via broadcast multiplication:
      Out = X * S.

Why this design:
- Very lightweight and easy to insert into CNN blocks.
- Emphasizes informative channels and suppresses less useful ones.
- Often yields consistent gains with minimal computational overhead.

ZH:
SE（Squeeze-and-Excitation）是一种通道注意力机制，通过
Squeeze（压缩）- Excitation（激励）- Scale（重标定）三步，
自适应调整各通道响应，显式建模通道间依赖关系。

整体结构：
1）Squeeze（压缩）
    - 输入 X 形状为 (N, C, H, W)。
    - 通过全局平均池化将空间信息压缩为通道描述向量：
      Z 形状为 (N, C, 1, 1)。

2）Excitation（激励）
    - 使用两层瓶颈变换（常称 MLP，这里用 1x1 卷积实现）：
      C -> C/r -> C。
    - 两层之间加入非线性激活（如 ReLU）。
    - 经过 sigmoid 得到范围在 [0, 1] 的通道权重 S，形状为 (N, C, 1, 1)。

3）Scale（重标定）
    - 将权重 S 通过广播机制乘回输入特征：Out = X * S。

设计动机：
- 结构轻量，易于插入各类卷积模块。
- 强化有用通道、抑制冗余通道。
- 计算开销小，通常能稳定提升模型性能。
"""

import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, channels:int , reduction:int = 10):# channels: 输入特征图的通道数, reduction: 隐藏层通道数的缩放因子(分母)
        super().__init__()
        assert channels > 0 # 输入通道数必须大于0
        hidden = max(1, channels//reduction) # 隐藏层通道数，至少为1,防止输入通道过少
        
        '''Squeeze: 全剧平均池化(B,C,H,W), -> (B,C,1,1)'''
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        '''2.两层通道MLP'''
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True) # ReLU激活函数，inplace=True表示直接在输入上进行操作，节省内存
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        
        '''3. Sigmoid激活函数'''
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        b,c,h,w = x.shape
        
        '''1. Squeeze'''
        z = self.gap(x)
        
        '''2.两层通道MLP'''
        s = self.fc1(z)
        s = self.act(s)
        s = self.fc2(s)
        
        '''3. Sigmoid激活函数'''
        s = self.sigmoid(s)
        
        '''4. Scale: 将输入特征图的每个通道乘以对应的权重'''
        out = x * s
        return out
        
        
if __name__ == "__main__":
    input = torch.randn(2, 16, 32, 32) # (B,C,H,W)
    se = SE(channels=16, reduction=4)
    output = se(input)
    print(output.shape) # (B,C,H,W) 与输入形状相同
"""
EN:
Triplet Attention builds lightweight attention across three pairwise dimension
interactions by rotating tensor axes and reusing the same gating mechanism.
Compared with standard spatial-only attention, it captures cross-dimension
dependencies among channel, height, and width with low overhead.

Core idea:
- Use three branches to model attention on different dimension pairs:
  1) H-W branch (standard spatial view)
  2) H-C branch (after axis permutation)
  3) W-C branch (after axis permutation)
- Aggregate branch outputs by averaging.

High-level structure:
1) AttentionGate design
    - For an input tensor, first apply ZPool:
      concatenate channel-wise max and mean maps -> (B, 2, H, W).
    - Then use Conv(2->1, kxk) + BN + sigmoid to produce an attention map.
    - Reweight input feature by element-wise multiplication.

2) Three-branch processing in TripletAttention
    - H-W branch: directly apply gate on X: (B, C, H, W).
    - H-C branch: permute X to (B, H, C, W), apply gate, then permute back.
    - W-C branch: permute X to (B, W, H, C), apply gate, then permute back.

3) Fusion
    - If no_spatial=False: average all three outputs.
    - If no_spatial=True: average only H-C and W-C outputs.
    - Final output keeps the same shape as input: (B, C, H, W).

Why this design:
- Captures interactions across different axis pairs without heavy modules.
- Reuses one simple gate design for multiple views.
- Provides stronger feature recalibration with modest extra computation.

ZH:
Triplet Attention 通过张量维度置换与共享门控结构，在三种“两两维度交互”视角下
执行轻量注意力建模。相比只在空间维建模的注意力，它能以较低开销同时捕捉
通道、高度、宽度之间的跨维依赖关系。

核心思想：
- 使用三个分支分别建模不同维度对：
  1）H-W 分支（标准空间视角）
  2）H-C 分支（通过置换后建模）
  3）W-C 分支（通过置换后建模）
- 最后对多个分支输出做平均融合。

整体结构：
1）AttentionGate 设计
    - 对输入先做 ZPool：
      将按通道最大值图与均值图拼接，得到 (B, 2, H, W)。
    - 再经 Conv(2->1, kxk) + BN + sigmoid 得到注意力图。
    - 与输入逐元素相乘完成重标定。

2）TripletAttention 三分支流程
    - H-W 分支：直接在 X（B, C, H, W）上做门控。
    - H-C 分支：将 X 置换为 (B, H, C, W)，门控后再置换回原布局。
    - W-C 分支：将 X 置换为 (B, W, H, C)，门控后再置换回原布局。

3）融合方式
    - no_spatial=False 时：三分支输出取平均。
    - no_spatial=True 时：仅融合 H-C 与 W-C 两分支。
    - 最终输出形状与输入一致： (B, C, H, W)。

设计动机：
- 在不引入重型结构的情况下建模跨轴交互。
- 共享简洁门控单元，参数与计算开销较低。
- 能在较小额外成本下增强特征重标定能力。
"""

import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ZPool(nn.Module):
    def forward(self,x):
        avg = x.mean(dim=1, keepdim=True)
        max = x.max(dim=1, keepdim=True)[0]
        return torch.cat((max, avg), dim=1)
    
class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.zpool = ZPool()
        padding = kernel_size // 2
        
        self.conv = BasicConv(2, 1, kernel_size=kernel_size, stride=1, padding=padding, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.zpool(x)
        z = self.conv(z)
        attn = self.sigmoid(z)
        return attn * x
        



class TripletAttention(nn.Module):
    def __init__(self, channels, kernel_size=7, no_spatial=False):
        super().__init__()
        self.channels = channels
        self.no_spatial = no_spatial
        
        # three gate
        self.gate_hw = AttentionGate(kernel_size=kernel_size)
        self.gate_hc = AttentionGate(kernel_size=kernel_size)
        self.gate_wc = AttentionGate(kernel_size=kernel_size) 
        
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3) # (B,H,C,W)
        x_perm2 = x.permute(0,3,2,1) # (B,W,C,H)
        

        out_hc = self.gate_hc(x_perm1).permute(0,2,1,3) # (B,C,H,W)
        out_wc = self.gate_wc(x_perm2).permute(0,2,1,3) # (B,C,H,W)
        
        if self.no_spatial:
            out = (out_wc + out_hc) / 2.0
        else:
            out_hw = self.gate_hw(x)
            out = (out_hw + out_hc + out_wc) / 3.0
        
        return out
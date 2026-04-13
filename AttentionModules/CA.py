"""
EN:
CA (Coordinate Attention) factorizes spatial attention into two 1D directions
(height and width) and embeds positional information into channel attention.
Compared with standard channel-only attention, CA can highlight important
regions while preserving long-range directional context.

High-level structure:
1) Directional pooling
     - Height branch: average over width, producing x_h with shape (N, C, H, 1).
     - Width branch: average over height, producing x_w with shape (N, C, 1, W),
         then permuted to (N, C, W, 1) for later concatenation.

2) Shared bottleneck transform
     - Concatenate x_h and transformed x_w along spatial dimension to obtain
         y with shape (N, C, H+W, 1).
     - Apply a shared 1x1 bottleneck MLP-like block (Conv-BN-Activation):
         C -> hidden, where hidden = max(8, C/reduction).

3) Split and projection
     - Split y back into y_h (N, hidden, H, 1) and y_w (N, hidden, W, 1).
     - Permute y_w to (N, hidden, 1, W), then use two independent 1x1 convs to
         project both branches from hidden -> C.

4) Attention generation and reweighting
     - Apply sigmoid on both branches to get directional attention maps:
         a_h: (N, C, H, 1), a_w: (N, C, 1, W).
     - Reweight input by broadcast multiplication: out = x * a_h * a_w.

Why this design:
- Encodes precise coordinate-aware cues using separate H/W dependencies.
- Keeps computation lightweight via 1x1 bottleneck operations.
- Preserves spatial resolution while improving feature selectivity.

ZH:
CA（Coordinate Attention，坐标注意力）将空间注意力分解为高度和宽度两个一维方向，
并把位置信息嵌入到通道注意力中。相比仅做通道建模的注意力，CA 能在保持方向上下文的同时，
更准确地强调关键区域。

整体结构：
1）方向池化
     - 高度分支：对宽度维做平均，得到 x_h，形状为 (N, C, H, 1)。
     - 宽度分支：对高度维做平均，得到 x_w，形状为 (N, C, 1, W)，
         再转置为 (N, C, W, 1) 以便拼接。

2）共享瓶颈变换
     - 在空间维拼接 x_h 与转置后的 x_w，得到 y，形状为 (N, C, H+W, 1)。
     - 经过共享的 1x1 瓶颈 MLP 风格模块（Conv-BN-Activation），
         通道从 C 压缩到 hidden，其中 hidden = max(8, C/reduction)。

3）拆分与回投影
     - 将 y 按空间维拆回 y_h (N, hidden, H, 1) 与 y_w (N, hidden, W, 1)。
     - y_w 转置回 (N, hidden, 1, W) 后，两个分支分别经过独立 1x1 卷积，
         将通道从 hidden 恢复到 C。

4）注意力生成与重标定
     - 两个分支经 sigmoid 得到方向注意力：
         a_h: (N, C, H, 1), a_w: (N, C, 1, W)。
     - 通过广播逐元素相乘完成重标定：out = x * a_h * a_w。

设计动机：
- 通过分方向建模引入更精确的坐标感知能力。
- 使用 1x1 瓶颈结构控制参数量和计算量。
- 不降采样空间分辨率，适合密集预测与细粒度表示任务。
"""

import torch
import torch.nn as nn
class CA(nn.Module):
    '''Coordinate Attention (CA) for enhanced feature representation.'''
    '''Coordinate Attention (CA)用于增强特征表示。'''
    '''CA通过在通道和空间维度上应用注意力机制，增强了模型关注重要特征的能力。'''
    
    
    
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        assert channels > 0
        hidden = max(8, channels//reduction)
        
        '''MLP'''
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        
        '''MLP->C'''
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b,c,h,w = x.shape
        
        # h,w做不同方向上的平均池化
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True) 
        
        # 将w方向的平均池化结果转置，使其与h方向的平均池化结果具有相同的形状
        x_w_T = x_w.permute(0,1,3,2) # 转置
        # 转置后便于concat操作，最终得到的y形状为(B,C,H+W,1)，其中H+W表示h和w方向上的平均池化结果拼接在一起
        y = torch.cat([x_h, x_w_T], dim=2) # (B,C,H+W,1)
        # 先经过一层1x1卷积将通道数从原始C压缩到自定义的Hidden，经过BatchNorm2d进行归一化，最后经过Hardswish激活函数引入非线性，得到的y形状为(B,Hidden,H+W,1)
        y = self.mlp(y)
        
        # 将融合到一起的自注意力图y分割成两个部分，分别对应h和w方向上的注意力权重，y_h的形状为(B,C,H,1)，y_w的形状为(B,C,W,1)
        y_h,y_w = torch.split(y, [h,w], dim=2) # (B,C,H,1), (B,C,W,1)
        
        y_w = y_w.permute(0,1,3,2) # 转置回原来的形状
        # 分别经过两层1x1卷积将通道数从Hidden恢复到原始的C，得到的y_h和y_w形状仍然为(B,C,H,1)和(B,C,W,1)
        y_h = self.conv_h(y_h)
        y_w = self.conv_w(y_w)
        # 最后经过Sigmoid激活函数将注意力权重压缩到0和1之间，得到的a_h和a_w形状仍然为(B,C,H,1)和(B,C,W,1)，表示h和w方向上的注意力权重
        a_h = self.sigmoid(y_h)
        a_w = self.sigmoid(y_w)
        
        out = x * a_h * a_w # 广播到同形状
        return out
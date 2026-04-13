"""
EN:
EMA (Efficient Multi-scale / Multi-axis Attention style block) performs attention
in grouped channel space and combines directional gating with local feature
interaction to refine representations efficiently.

Core idea:
- Split channels into groups to reduce computation.
- Build axis-aware gates from height/width pooled descriptors.
- Mix gated features with a local 3x3 branch, then compute spatial weights
  for adaptive reweighting.

High-level structure:
1) Grouped reshape
    - Input X: (B, C, H, W).
    - Reshape into group space: (B*G, Cg, H, W), where Cg = C/G.
    - Grouped processing lowers cost while keeping channel diversity.

2) Direction-aware gating (H/W branches)
    - Height descriptor: average over width -> (B*G, Cg, H, 1) style tensor.
    - Width descriptor: average over height -> (B*G, Cg, 1, W) style tensor.
    - Concatenate descriptors, apply 1x1 transform + activation, then split back.
    - Two separate 1x1 projections + sigmoid generate gates A_h and A_w.
    - Apply multiplicative gating: X1 = Xg * A_h * A_w.

3) Dual feature branches
    - Branch-1: normalized gated feature (GroupNorm in group space): X2.
    - Branch-2: local interaction via 3x3 conv + activation on original grouped
      feature: X3.

4) Spatial weighting and fusion
    - Flatten spatial dimensions and compute softmax-based spatial responses.
    - Use global summaries from one branch to modulate the other branch response
      (cross-branch interaction), then normalize to obtain spatial weight map S.
    - Final grouped output: Out_g = Xg * S, then reshape back to (B, C, H, W).

Why this design:
- Grouped computation improves efficiency on high-channel features.
- Directional gates inject long-range axis-aware context.
- Local 3x3 branch preserves neighborhood modeling ability.
- Cross-branch weighting improves complementarity between global and local cues.

ZH:
EMA（高效多尺度/多轴注意力风格模块）在“分组通道空间”内执行注意力建模，
将方向门控与局部特征交互结合，以较低开销增强特征表示。

核心思想：
- 先做通道分组，降低计算量；
- 利用高/宽两个方向的统计描述构建轴向门控；
- 再结合局部 3x3 分支与交互加权得到空间权重，完成重标定。

整体结构：
1）分组重排
    - 输入 X 形状为 (B, C, H, W)。
    - 重排到分组空间 (B*G, Cg, H, W)，其中 Cg = C/G。
    - 在分组内处理可降低复杂度，同时保留通道多样性。

2）方向感知门控（H/W 分支）
    - 高度描述：沿宽度维平均，得到类似 (B*G, Cg, H, 1) 的描述。
    - 宽度描述：沿高度维平均，得到类似 (B*G, Cg, 1, W) 的描述。
    - 拼接后经 1x1 变换与激活，再拆分回两支。
    - 两个独立 1x1 投影 + sigmoid 生成门控 A_h、A_w。
    - 进行逐元素门控：X1 = Xg * A_h * A_w。

3）双分支特征建模
    - 分支1：对门控特征做归一化（组空间内 GroupNorm），得到 X2。
    - 分支2：对原分组特征做 3x3 卷积 + 激活，得到 X3（建模局部交互）。

4）空间加权与融合
    - 展平空间维后计算 softmax 形式的空间响应。
    - 使用一支路的全局摘要去调制另一支路，实现交叉分支交互，
      再归一化得到空间权重图 S。
    - 最终分组输出为 Out_g = Xg * S，并重排回 (B, C, H, W)。

设计动机：
- 分组处理提升高通道特征下的效率。
- 方向门控引入长程轴向上下文信息。
- 3x3 局部分支保留邻域建模能力。
- 交叉分支加权提升全局与局部信息互补性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA(nn.Module):
    def __init__(self, channels, groups=32, act: nn.Module | None = None):
        super().__init__()
        self.channels = channels
        
        g = min(groups, channels)
        while g > 1 and channels % g != 0:
            g -= 1
        self.groups = max(1, g)
        self.cg = channels // self.groups
        
        self.act = act if act is not None else nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(self.cg, self.cg, 1, stride=1, padding=0, bias=True)
        self.conv_h = nn.Conv2d(self.cg, self.cg, 1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(self.cg, self.cg, 1, stride=1, padding=0, bias=True)
        
        # 3x3 branch (local intertactions)
        self.conv3 = nn.Conv2d(self.cg, self.cg, 3, stride=1, padding=1, bias=True)
        
        # Normalization in group space
        self.gn = nn.GroupNorm(num_groups=1, num_channels=self.cg)
        
        self.sigmoid = nn.Sigmoid()
        
        self.softmax_spatial = lambda t: F.softmax(t, dim=-1) # spatial attention along H*W
        
    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.channels, f"Expected input with {self.channels} channels, got {c}"
        
        g = self.groups
        cg = self.cg
        hw = h * w
        
        xg = x.view(b, g, cg, h, w).view(b * g, cg, h, w) # (b*g,cg,h,w)
        
        x_h = xg.mean(dim=3, keepdim=True) # (b*g,cg,1,w)
        x_w = xg.mean(dim=2, keepdim=True).permute(0, 1, 3, 2) # (b*g,cg,h,1)
        
        y = torch.cat([x_h, x_w], dim=2) # (b*g,cg,h+w,1)
        y = self.conv1(y) # (b*g,cg,h+w,1)
        y = self.act(y)
        
        y_h, y_w = torch.split(y, [h, w], dim=2) # (b*g,cg,h,1), (b*g,cg,w,1)
        y_w = y_w.permute(0, 1, 3, 2) # (b*g,cg,1,w)
        a_h = self.sigmoid(self.conv_h(y_h)) # (b*g,cg,h,1)
        a_w = self.sigmoid(self.conv_w(y_w)) # (b*g,cg,1,w)
        
        x1 = xg * a_h * a_w # (b*g,cg,h,w)
        x2 = self.gn(x1) # (b*g,cg,h,w)
        x3 = self.act(self.conv3(xg)) # (b*g,cg,h,w) 
        
        A2 = self.softmax_spatial(x2.view(b*g, cg, hw)) # (b,g,cg,hw)
        A3 = self.softmax_spatial(x3.view(b*g, cg, hw)) # (b,g,cg,hw)
        
        V2 = x2.mean(dim=(2, 3)).unsqueeze(1) # (b*g,cg,1)
        V3 = x3.mean(dim=(2, 3)).unsqueeze(1) # (b*g,cg,1)
        
        s2 = torch.bmm(V2, A3.view(b*g, 1, h, w))
        s3 = torch.bmm(V3, A2.view(b*g, 1, h, w))
        s = self.softmax_spatial(s2 + s3) # (b*g,1,h,w)
        
        out_g = xg * s # (b*g,cg,h,w)
        out = out_g.view(b, g, cg, h, w).reshape(b, c, h, w) # (b,c,h,w)
        
        return out
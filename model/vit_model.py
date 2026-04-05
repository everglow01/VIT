"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn


# Patch Embedding
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    将二维图像切成不重叠 patch，并映射到 embedding 维度 D，输出 token 序列
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768,
                 norm_layer: Optional[Callable[[int], nn.Module]] = None):
        super().__init__()

        # 统一成 (H, W) 和 (P, P) 形式，便于后续计算
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size   # 输入图像尺寸，例如 (224, 224)
        self.patch_size = patch_size  # patch 尺寸，例如 (16, 16)

        # 网格大小：一行多少个 patch、一列多少个 patch
        # 例如 224/16=14，因此 grid_size=(14,14)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        # patch 总数量 N = 14*14 = 196
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 关键：用 Conv2d 等价实现 “切 patch + 线性投影”
        # 输入通道 in_c=3，输出通道 embed_dim=D
        # kernel=stride=patch_size => 不重叠地覆盖每个 patch
        # 输出特征图形状：[B, D, H/P, W/P]，例如 [B,768,14,14]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 可选的归一化层（有些实现会在 patch embedding 后加 LN/BN）
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W] 例如 [B,3,224,224]
        B, C, H, W = x.shape

        # ViT原始实现通常固定训练分辨率
        # 如果想支持任意分辨率，这里一般不 assert，而是动态计算grid_size，并对pos_embed插值
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # 1) proj： [B,3,224,224] -> [B,768,14,14]
        # 2) flatten(2)：把空间维展平
        #    [B,768,14,14] -> [B,768,196]
        # 3) transpose(1,2)：把 token 维度放到中间，得到序列形式
        #    [B,768,196] -> [B,196,768]
        x = self.proj(x).flatten(2).transpose(1, 2)

        # norm：保持形状不变 [B,196,768]
        x = self.norm(x)
        return x

# 多头注意力机制
class Attention(nn.Module):
    def __init__(self,
                 dim,                 # token embedding 维度 C（例如 768）
                 num_heads=8,         # 多头数 h
                 qkv_bias=False,      # qkv 线性层是否带 bias
                 qk_scale=None,       # 可选：自定义缩放因子，默认 1/sqrt(head_dim)
                 attn_drop_ratio=0.,  # attention 权重 dropout
                 proj_drop_ratio=0.): # 输出投影 dropout
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads           # 每个头的维度 d = C/h
        self.scale = qk_scale or head_dim ** -0.5  # 缩放：1/sqrt(d)，避免 QK^T 数值过大

        # 一次线性层同时生成 Q,K,V（更高效）
        # 输入 [B,N,C] -> 输出 [B,N,3C]
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

        # 多头 concat 后再做一次输出投影 Wo
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x: [B, N, C]  N = patch数 + 1（cls），C = dim
        B, N, C = x.shape

        # 1) 生成 qkv: [B,N,3C]
        # 2) reshape: [B,N,3,h,d]  (d = C/h)
        # 3) permute: [3,B,h,N,d]  方便拆出 q,k,v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q,k,v: [B,h,N,d]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 加权求和得到每个 token 的新表示：
        # attn @ v: [B,h,N,d]
        # transpose: [B,N,h,d]
        # reshape: [B,N,C]  (把多头拼回去)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 缩放
        attn = attn.softmax(dim=-1)                    # 对最后一维 N 做 softmax
        attn = self.attn_drop(attn)                    # dropout on attention weights

        # 输出投影（对应 Wo）：[B,N,C] -> [B,N,C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    ViT / Transformer Encoder Block 中的前馈网络（FFN/MLP）
    结构：Linear -> GELU -> Dropout -> Linear -> Dropout
    注意：对每个 token 独立作用，不改变 token 数 N
    """

    def __init__(self,
                 in_features,  # 输入维度 C（例如 768）
                 hidden_features=None,  # 隐藏层维度（通常是 C * mlp_ratio，如 4C=3072）
                 out_features=None,  # 输出维度，默认等于输入维度
                 act_layer=nn.GELU,  # 激活函数，ViT 默认用 GELU
                 drop=0.):  # dropout 概率（通常叫 proj_drop / mlp_drop）
        super().__init__()

        # 默认保持输入输出维度一致（残差连接要求）
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 第1层：升维（C -> hidden）
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活：GELU（ViT/BERT常用）
        self.act = act_layer()
        # 第2层：降维（hidden -> C）
        self.fc2 = nn.Linear(hidden_features, out_features)
        # dropout：在 fc1 后和 fc2 后都会用一次
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, N, C]
        x = self.fc1(x)  # [B, N, C] -> [B, N, hidden]
        x = self.act(x)  # 非线性
        x = self.drop(x)  # 正则化

        x = self.fc2(x)  # [B, N, hidden] -> [B, N, C]
        x = self.drop(x)  # 正则化
        return x


class Block(nn.Module):
    """
       ViT / Transformer Encoder的一个Block（Pre-LN）
       结构：
        x = x + MSA(LN(x))
        x = x + MLP(LN(x))
       输入输出形状不变：[B, N, D]
    """
    def __init__(self,
                 dim,  # token维度 D（如 768）
                 num_heads,  # 注意力头数 h
                 mlp_ratio=4.,  # MLP隐藏层扩展比例（hidden = D*mlp_ratio）
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,  # MLP输出 / attention输出 的 dropout
                 attn_drop_ratio=0.,  # attention权重 dropout
                 act_layer: Callable[[], nn.Module] = nn.GELU,
                 norm_layer: Callable[[int], nn.Module] = nn.LayerNorm):
        super(Block, self).__init__()

        # 1) 第一个 LN：给 Attention 前做归一化（Pre-LN）
        self.norm1 = norm_layer(dim)

        # 2) 多头自注意力：输入输出都是 [B,N,D]
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # 3) 第二个LN：给MLP前做归一化（Pre-LN）
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # x: [B, N, D]
        # ---- Attention 子层 ----
        # LN: [B,N,D] -> [B,N,D]
        # MSA: [B,N,D] -> [B,N,D]
        # Residual: x + branch
        x = x + self.attn(self.norm1(x))

        # ---- MLP 子层 ----
        # LN: [B,N,D] -> [B,N,D]
        # MLP: [B,N,D] -> [B,N,D]
        x = x + self.mlp(self.norm2(x))
        return x


def _init_vit_weights(m):
    """
    ViT 权重初始化（根据层类型分别初始化）
    m: 传入的子模块（apply 会遍历模型里所有子模块）
    """
    # 1) 线性层 Linear：trunc_normal 初始化权重，bias 全 0
    #   - trunc_normal：截断正态分布，避免极端大值，Transformer 中很常用
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)   # 权重：N(0,0.01) 的截断版本
        if m.bias is not None:
            nn.init.zeros_(m.bias)                 # 偏置：0

    # 2) 卷积层 Conv2d：Kaiming 初始化（更适合 ReLU/卷积类）
    #   - 这里主要影响 PatchEmbed 那个 Conv2d（以及你若加 CNN stem）
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # 3) LayerNorm：weight=1，bias=0（保证一开始是“标准化但不缩放/偏移”）
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0.,  embed_layer=PatchEmbed,
                 norm_layer: Optional[Callable[[int], nn.Module]] = None,
                 act_layer: Optional[Callable[[], nn.Module]] = None):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 这里num_features主要给分类头用（后面可能被 representation_size 改写）

        # 默认LayerNorm（eps=1e-6）和 GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        if representation_size is None:
            representation_size = embed_dim

        # 1) Patch Embedding：图像 -> patch tokens
        # 输出：[B, N, D]，N=196（224/16=14, 14*14=196）
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2) 可学习 cls token：shape [1,1,D]，forward 时 expand 成 [B,1,D]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3) 可学习绝对位置编码：shape [1, N + num_tokens, D]
        # 标准 ViT: [1, 197, 768]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 对加了位置编码后的序列做dropout（正则）
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 5) Transformer Encoder：堆叠 depth 个 Block
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)])

        # 最后再做一次 LayerNorm（论文也有最后的 LN）
        self.norm = norm_layer(embed_dim)

        # 加一层 Linear + Tanh
        self.num_features = representation_size
        self.pre_logits = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(embed_dim, representation_size)),
            ("act", nn.Tanh())
        ]))

        # 分类头：把特征映射到类别数 K
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init：pos_embed / cls_token / dist_token 采用 trunc_normal(std=0.02)（常见 ViT 初始化）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 对其它层按自定义 _init_vit_weights 初始化
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, N, D]
        x = self.patch_embed(x)

        # [1, 1, D] -> [B, 1, D]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # [B, N+1, D]
        x = torch.cat((cls_token, x), dim=1)

        # 加位置编码
        x = self.pos_drop(x + self.pos_embed)

        # Encoder
        x = self.blocks(x)
        x = self.norm(x)

        # 取 cls
        return self.pre_logits(x[:, 0])

    def forward_features_map(self, x):
        """
        返回 patch tokens reshape 成 2D feature map，供检测/分割 neck 使用。
        输出: [B, embed_dim, H/patch_size, W/patch_size]，例如 [B, 768, 14, 14]
        """
        x = self.patch_embed(x)                              # [B, N, D]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)                 # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # 去掉 cls token，reshape 成 2D
        patch_tokens = x[:, 1:]                              # [B, N, D]
        H = W = self.patch_embed.grid_size[0]
        B, _, D = patch_tokens.shape
        return patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_base_patch16_224_in21k(num_classes: int = 21843):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/loadreleases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280,
                              num_classes=num_classes)
    return model

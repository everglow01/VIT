"""
Swin Transformer backbone.
Adapted from timm (https://github.com/huggingface/pytorch-image-models) — no timm dependency.

Exposed API (mirrors vit_model.py convention):
  SwinTransformer:
    forward(x)               -> [B, num_classes]
    forward_features_map(x)  -> dict{"p2":[B,C,H/4,W/4], "p3":..., "p4":..., "p5":...}
    embed_dim                -> int (stage-1 channels, e.g. 96)

Factory functions:
  swin_tiny_patch4_window7_224()
  swin_small_patch4_window7_224()
  swin_base_patch4_window7_224()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _to_2tuple(x):
    return (x, x) if not isinstance(x, (list, tuple)) else tuple(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x, window_size: int):
    """[B, H, W, C] -> [num_windows*B, window_size, window_size, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size: int, H: int, W: int):
    """[num_windows*B, window_size, window_size, C] -> [B, H, W, C]"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(self.window_size[0] * self.window_size[1],
                       self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path_rate=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # Precompute mask for the default (init-time) resolution
        if self.shift_size > 0:
            attn_mask = self._compute_attn_mask(
                *self.input_resolution, self.window_size, self.shift_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    @staticmethod
    def _compute_attn_mask(H, W, window_size, shift_size):
        """Build shifted-window attention mask for the given spatial size."""
        img_mask = torch.zeros(1, H, W, 1)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x, H, W):
        B, L, C = x.shape
        ws = self.window_size
        ss = self.shift_size

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiples of window_size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Shifted window attention
        if ss > 0:
            x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))
            # Reuse precomputed mask when padded size matches init resolution
            if (Hp, Wp) == tuple(self.input_resolution):
                mask = self.attn_mask
            else:
                mask = self._compute_attn_mask(Hp, Wp, ws, ss).to(x.device)
        else:
            mask = None

        x_windows = window_partition(x, ws).view(-1, ws * ws, C)
        attn_windows = self.attn(x_windows, mask=mask)
        x = window_reverse(attn_windows.view(-1, ws, ws, C), ws, Hp, Wp)

        if ss > 0:
            x = torch.roll(x, shifts=(ss, ss), dims=(1, 2))

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        # Pad if H or W is odd so 2x2 merging works
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                        x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], dim=-1)
        H_out, W_out = (H + 1) // 2, (W + 1) // 2
        return self.reduction(self.norm(x.view(B, -1, 4 * C))), H_out, W_out


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
                drop_path_rate=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        before_downsample = x
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, before_downsample, H, W


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                          # [B, C, H_grid, W_grid]
        H, W = x.shape[2], x.shape[3]
        x = self.norm(x.flatten(2).transpose(1, 2))  # [B, H*W, C]
        return x, H, W


class SwinTransformer(nn.Module):
    """
    Swin Transformer backbone.

    forward_features_map(x) returns:
        {"p2": [B, C,   H/4,  W/4],
         "p3": [B, 2C,  H/8,  W/8],
         "p4": [B, 4C,  H/16, W/16],
         "p5": [B, 8C,  H/32, W/32]}
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                   patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        # per-stage norms for feature map extraction
        for i in range(self.num_layers):
            self.add_module(f"norm{i}", norm_layer(int(embed_dim * 2 ** i)))

        self.norm_cls = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes) \
            if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features_map(self, x):
        """Returns dict of 4 feature maps keyed 'p2'..'p5'.

        Supports arbitrary input resolutions — spatial dimensions are computed
        dynamically from the actual tensor rather than the init-time img_size.
        """
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        feature_maps = {}
        for i, layer in enumerate(self.layers):
            H_in, W_in = H, W          # resolution entering this stage
            x, before_down, H, W = layer(x, H_in, W_in)
            norm = getattr(self, f"norm{i}")
            feat = norm(before_down)
            C = feat.shape[-1]
            feature_maps[f"p{i+2}"] = feat.transpose(1, 2).reshape(-1, C, H_in, W_in)
        return feature_maps

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, _, H, W = layer(x, H, W)
        x = self.norm_cls(x)
        x = torch.flatten(self.avgpool(x.transpose(1, 2)), 1)
        return self.head(x)


def swin_tiny_patch4_window7_224(num_classes: int = 1000) -> SwinTransformer:
    """Swin-Tiny: embed_dim=96, depths=[2,2,6,2], stage channels: 96/192/384/768"""
    return SwinTransformer(img_size=224, patch_size=4, embed_dim=96,
                           depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                           window_size=7, drop_path_rate=0.2, num_classes=num_classes)


def swin_small_patch4_window7_224(num_classes: int = 1000) -> SwinTransformer:
    """Swin-Small: embed_dim=96, depths=[2,2,18,2], stage channels: 96/192/384/768"""
    return SwinTransformer(img_size=224, patch_size=4, embed_dim=96,
                           depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
                           window_size=7, drop_path_rate=0.3, num_classes=num_classes)


def swin_base_patch4_window7_224(num_classes: int = 1000) -> SwinTransformer:
    """Swin-Base: embed_dim=128, depths=[2,2,18,2], stage channels: 128/256/512/1024"""
    return SwinTransformer(img_size=224, patch_size=4, embed_dim=128,
                           depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                           window_size=7, drop_path_rate=0.5, num_classes=num_classes)

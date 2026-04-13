"""
Mask prediction branch for DETR-style instance segmentation.

Takes decoder query features and encoder memory, produces per-instance
mask logits at p3 resolution (H/8 × W/8 for 224 input).
"""
import torch
import torch.nn as nn
from AttentionModules.DANet import DANet


class MaskBranch(nn.Module):
    """
    Predicts a mask for each object query via dot product between
    projected query embeddings and projected encoder feature map.

    Uses the p3-level slice of the encoder memory (highest resolution)
    so output is already at H_img/8 × W_img/8 — no extra upsampling needed.
    """
    def __init__(self, d_model=256, mask_dim=256):
        super().__init__()
        self.query_proj = nn.Linear(d_model, mask_dim)
        self.memory_proj = nn.Conv2d(d_model, mask_dim, kernel_size=1)
        self.danet = DANet(mask_dim)

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[MaskBranch] params: {total:,}")

    def forward(self, hs, memory, spatial_shapes):
        """
        Args:
            hs:             [B, N, d_model]  decoder query features
            memory:         [B, HW_total, d_model]  encoder output
            spatial_shapes: list of (H, W) for [p3, p4, p5]

        Returns:
            masks: [B, N, H_p3, W_p3]  logits (no sigmoid)
        """
        H3, W3 = spatial_shapes[0]          # p3 is the first (highest-res) level
        p3_len = H3 * W3

        # Extract p3-level memory and reshape to spatial
        memory_p3 = memory[:, :p3_len]                         # [B, H3*W3, d_model]
        B = memory_p3.shape[0]
        feat_map = memory_p3.transpose(1, 2).reshape(B, -1, H3, W3)  # [B, d_model, H3, W3]
        feat_map = self.memory_proj(feat_map)                  # [B, mask_dim, H3, W3]
        feat_map = self.danet(feat_map)                        # [B, mask_dim, H3, W3]

        # Project queries
        mask_emb = self.query_proj(hs)                         # [B, N, mask_dim]

        # Dot product → per-query masks
        masks = torch.einsum("bnd,bdhw->bnhw", mask_emb, feat_map)
        return masks  # [B, N, H3, W3]

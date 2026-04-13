"""
DN-DETR style Transformer Encoder-Decoder head.

Accepts multi-scale Swin features (p3, p4, p5), runs a Transformer
encoder-decoder, and optionally constructs denoising queries from GT
during training (DN-DETR).

Output dict contains pred_logits, pred_boxes, hs (query features for
mask branch), memory (encoder output), and DN predictions when training.
"""
import math
import torch
import torch.nn as nn
from AttentionModules.SCSA import SCSA


# ======================================================================
# Positional Encoding
# ======================================================================

class PositionEmbeddingSine2D(nn.Module):
    """Non-learnable 2-D sinusoidal positional encoding."""
    def __init__(self, d_model=256, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, H, W, device):
        """Returns [HW, d_model] positional embedding."""
        half_d = self.d_model // 2

        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).expand(H, W)

        # Normalise to [0, 1]
        y_embed = y_embed / max(H - 1, 1)
        x_embed = x_embed / max(W - 1, 1)

        dim_t = torch.arange(half_d, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_d)

        pos_x = x_embed.unsqueeze(-1) / dim_t          # [H, W, half_d]
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1)        # [H, W, d_model]
        return pos.flatten(0, 1)                        # [HW, d_model]


# ======================================================================
# Feed-forward helpers
# ======================================================================

class MLP(nn.Module):
    """k-layer MLP (used for bbox regression head)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        ])
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = torch.relu(x)
        return x


# ======================================================================
# DN-DETR denoising query construction
# ======================================================================

def prepare_dn_components(targets, num_queries, d_model, num_classes,
                          label_enc, num_dn_groups=2, noise_scale=0.4,
                          label_noise_ratio=0.5):
    """
    Build denoising queries from GT targets (basic DN-DETR, positive only).

    For each DN group, create max_M queries initialised from noised GT boxes
    and (optionally) noised GT labels. All are supervised by the original GT.

    Args:
        targets:  list[dict] with 'labels' [M] (0-based) and 'boxes' [M, 4]
        num_queries:  number of matching queries
        d_model:  embedding dimension
        num_classes:  foreground class count
        label_enc:  nn.Embedding(num_classes + 1, d_model)
        num_dn_groups:  number of independent denoising groups
        noise_scale:  box noise magnitude relative to box size
        label_noise_ratio:  probability of flipping a label to a random class

    Returns:
        dn_queries:   [B, dn_len, d_model]
        dn_attn_mask: [total, total]  (float, 0 / -inf)
        dn_meta:      dict for loss computation
        Returns (None, None, None) if no GT objects in the batch.
    """
    B = len(targets)
    device = label_enc.weight.device

    num_gts = [len(t["labels"]) for t in targets]
    max_M = max(num_gts) if num_gts else 0
    if max_M == 0:
        return None, None, None

    dn_len = num_dn_groups * max_M
    total = num_queries + dn_len

    # Pad GT to max_M per image
    gt_labels_pad = torch.full((B, max_M), num_classes, dtype=torch.int64, device=device)
    gt_boxes_pad = torch.zeros((B, max_M, 4), dtype=torch.float32, device=device)
    gt_valid = torch.zeros((B, max_M), dtype=torch.bool, device=device)

    for b in range(B):
        M = num_gts[b]
        if M > 0:
            gt_labels_pad[b, :M] = targets[b]["labels"]
            gt_boxes_pad[b, :M] = targets[b]["boxes"]
            gt_valid[b, :M] = True

    # Build DN queries per group (with increasing noise level)
    all_dn_labels = []
    all_dn_valid = []

    for g in range(num_dn_groups):
        scale = noise_scale * (1.0 + g * 0.3)   # slightly more noise per group

        # Noised labels
        labels_g = gt_labels_pad.clone()
        if label_noise_ratio > 0:
            flip_mask = torch.rand(B, max_M, device=device) < label_noise_ratio
            flip_mask = flip_mask & gt_valid
            random_labels = torch.randint(0, num_classes, (B, max_M), device=device)
            labels_g[flip_mask] = random_labels[flip_mask]

        # Noised boxes
        boxes_g = gt_boxes_pad.clone()
        box_wh = boxes_g[..., 2:4].clamp(min=1e-4)
        noise = (torch.rand_like(boxes_g) * 2 - 1)           # [-1, 1]
        noise[..., :2] *= box_wh * scale
        noise[..., 2:4] *= box_wh * scale
        boxes_g = (boxes_g + noise).clamp(0, 1)

        all_dn_labels.append(labels_g)
        all_dn_valid.append(gt_valid)

    dn_labels = torch.cat(all_dn_labels, dim=1)    # [B, dn_len]
    dn_valid = torch.cat(all_dn_valid, dim=1)      # [B, dn_len]

    # Content embedding from (noised) labels
    dn_queries = label_enc(dn_labels)               # [B, dn_len, d_model]

    # ---- Attention mask ----
    # Start with everything masked, then open allowed pairs
    attn_mask = torch.full((total, total), float("-inf"), device=device)

    # Matching queries can attend to each other
    attn_mask[:num_queries, :num_queries] = 0

    # Each DN group can attend within itself and to matching queries
    for g in range(num_dn_groups):
        s = num_queries + g * max_M
        e = s + max_M
        attn_mask[s:e, s:e] = 0                # intra-group
        attn_mask[s:e, :num_queries] = 0       # DN → matching

    # ---- DN supervision targets ----
    dn_gt_labels = torch.cat([gt_labels_pad] * num_dn_groups, dim=1)  # [B, dn_len]
    dn_gt_boxes = torch.cat([gt_boxes_pad] * num_dn_groups, dim=1)    # [B, dn_len, 4]

    dn_meta = {
        "num_groups": num_dn_groups,
        "max_M": max_M,
        "valid_mask": dn_valid.float(),
        "gt_labels": dn_gt_labels,
        "gt_boxes": dn_gt_boxes,
    }

    return dn_queries, attn_mask, dn_meta


# ======================================================================
# DETRHead
# ======================================================================

class DETRHead(nn.Module):
    """
    Transformer Encoder-Decoder with multi-scale input and DN-DETR denoising.

    Takes Swin features {p3, p4, p5}, projects + flattens + encodes them,
    then decodes with learnable object queries (+ DN queries during training).
    """
    def __init__(
        self,
        in_channels_list,       # [C_p3, C_p4, C_p5]
        num_classes,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_queries=100,
        dropout=0.1,
        num_dn_groups=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_dn_groups = num_dn_groups

        # Per-level input projection
        self.input_projs = nn.ModuleList([
            nn.Linear(c, d_model) for c in in_channels_list
        ])

        # Per-level SCSA after projection (spatial attention before encoder)
        self.scsa_levels = nn.ModuleList([
            SCSA(d_model) for _ in in_channels_list
        ])

        # Learnable level embedding to distinguish scales
        self.level_embed = nn.Embedding(len(in_channels_list), d_model)

        # Shared positional encoding
        self.pos_enc = PositionEmbeddingSine2D(d_model)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)   # content
        self.query_pos = nn.Embedding(num_queries, d_model)     # positional

        # DN label embedding (shared with denoising query construction)
        self.label_enc = nn.Embedding(num_classes + 1, d_model)

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

        self._reset_parameters()
        self._print_params()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.query_embed.weight)
        nn.init.normal_(self.query_pos.weight)

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DETRHead] params: {total:,} total, {trainable:,} trainable")

    def forward(self, features, targets=None):
        """
        Args:
            features: dict  {p3: [B,C3,H3,W3], p4: [B,C4,H4,W4], p5: [B,C5,H5,W5]}
            targets:  list[dict] with 0-based labels & normalised cxcywh boxes
                      (only used for DN queries during training)
        Returns:
            dict with keys:
              pred_logits    [B, nq, num_classes+1]
              pred_boxes     [B, nq, 4]          cxcywh normalised
              hs             [B, nq, d_model]    decoder query features
              memory         [B, HW_total, d_model]
              spatial_shapes list[(H,W), ...]
              (training only)
              dn_pred_logits [B, dn_len, num_classes+1]
              dn_pred_boxes  [B, dn_len, 4]
              dn_meta        dict
        """
        device = next(iter(features.values())).device

        # ---------- Multi-scale flatten + project + PE ----------
        src_list = []
        spatial_shapes = []
        for i, key in enumerate(["p3", "p4", "p5"]):
            feat = features[key]                        # [B, C, H, W]
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))

            feat_flat = feat.flatten(2).transpose(1, 2) # [B, HW, C]
            feat_proj = self.input_projs[i](feat_flat)  # [B, HW, d_model]

            # SCSA: reshape to spatial → attend → flatten back
            feat_proj = self.scsa_levels[i](
                feat_proj.transpose(1, 2).view(B, self.d_model, H, W)
            ).flatten(2).transpose(1, 2)                # [B, HW, d_model]

            pos = self.pos_enc(H, W, device)            # [HW, d_model]
            lvl = self.level_embed.weight[i]            # [d_model]
            feat_proj = feat_proj + pos.unsqueeze(0) + lvl.unsqueeze(0).unsqueeze(0)

            src_list.append(feat_proj)

        src = torch.cat(src_list, dim=1)                # [B, sum(HiWi), d_model]

        # ---------- Encoder ----------
        memory = self.encoder(src)                      # [B, HW_total, d_model]

        # ---------- Decoder queries ----------
        tgt = (self.query_embed.weight + self.query_pos.weight)  # [nq, d_model]
        tgt = tgt.unsqueeze(0).expand(B, -1, -1)                # [B, nq, d_model]

        # DN-DETR denoising queries (training only)
        dn_meta = None
        attn_mask = None
        if self.training and targets is not None:
            dn_queries, attn_mask, dn_meta = prepare_dn_components(
                targets, self.num_queries, self.d_model,
                self.num_classes, self.label_enc, self.num_dn_groups,
            )
            if dn_queries is not None:
                tgt = torch.cat([tgt, dn_queries], dim=1)

        # ---------- Decoder ----------
        hs = self.decoder(tgt, memory, tgt_mask=attn_mask)  # [B, nq+dn_len, d_model]

        # ---------- Split & predict ----------
        hs_matching = hs[:, :self.num_queries]

        pred_logits = self.class_embed(hs_matching)             # [B, nq, C+1]
        pred_boxes = self.bbox_embed(hs_matching).sigmoid()     # [B, nq, 4]

        out = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "hs": hs_matching,
            "memory": memory,
            "spatial_shapes": spatial_shapes,
        }

        if dn_meta is not None:
            hs_dn = hs[:, self.num_queries:]
            out["dn_pred_logits"] = self.class_embed(hs_dn)
            out["dn_pred_boxes"] = self.bbox_embed(hs_dn).sigmoid()
            out["dn_meta"] = dn_meta

        return out

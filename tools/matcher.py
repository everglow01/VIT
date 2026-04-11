"""
Hungarian bipartite matching for DETR-style set prediction.
Provides box format utilities and the HungarianMatcher used by SetCriterion.
"""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """Convert (x1, y1, x2, y2) -> (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes1, boxes2):
    """
    Pairwise IoU between two sets of boxes in xyxy format.
    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]
    Returns:
        iou:   [N, M]
        union: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    GIoU between two sets of boxes in xyxy format.
    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]
    Returns:
        giou: [N, M]
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # enclosing box area

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between predictions and ground truth.
    Cost = cost_class * C_cls + cost_bbox * C_L1 + cost_giou * C_giou.
    """
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits: [B, N, num_classes+1]
            pred_boxes:  [B, N, 4]  cx,cy,w,h normalised
            targets:     list[dict] each with 'labels' [M] and 'boxes' [M, 4]
        Returns:
            list of (pred_indices, gt_indices) tuples, one per image
        """
        B, N = pred_logits.shape[:2]
        indices = []

        for b in range(B):
            tgt_labels = targets[b]["labels"]  # [M]
            tgt_boxes = targets[b]["boxes"]    # [M, 4]
            M = len(tgt_labels)

            if M == 0:
                indices.append((torch.tensor([], dtype=torch.int64),
                                torch.tensor([], dtype=torch.int64)))
                continue

            # Classification cost: negative softmax probability of target class
            prob = pred_logits[b].softmax(-1)        # [N, C+1]
            cost_class = -prob[:, tgt_labels]        # [N, M]

            # L1 box cost
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # [N, M]

            # GIoU cost
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)   # [N, M]

            C = (self.cost_class * cost_class
                 + self.cost_bbox * cost_bbox
                 + self.cost_giou * cost_giou)

            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices

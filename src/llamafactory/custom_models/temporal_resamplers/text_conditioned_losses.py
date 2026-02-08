"""
Losses for text-conditioned temporal queries.
Simpler than slot matching - direct supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TextConditionedLoss(nn.Module):
    """
    Direct supervision - no Hungarian matching needed!
    Query 0 = interacted object → supervise with GT object boxes
    Query 1 = tool → supervise with GT tool boxes
    etc.
    """
    
    def __init__(
        self,
        box_weight: float = 2.0,
        giou_weight: float = 1.0,
        conf_weight: float = 1.0,
        temporal_weight: float = 0.5,
    ):
        super().__init__()
        self.box_weight = box_weight
        self.giou_weight = giou_weight
        self.conf_weight = conf_weight
        self.temporal_weight = temporal_weight
        
    def forward(
        self,
        output: Dict,
        gt_object_boxes: torch.Tensor,      # (B, T, 4)
        gt_tool_boxes: torch.Tensor,        # (B, T, 4)
        gt_robot_boxes: torch.Tensor,       # (B, T, 4)
        gt_start_location: torch.Tensor,    # (B, 4)
        gt_end_location: torch.Tensor,      # (B, 4)
        gt_target: torch.Tensor = None,     # (B, 4) optional
        gt_object_visible: torch.Tensor = None,  # (B, T) optional visibility mask
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # === Dynamic query losses (tracked) ===
        pred_dynamic = output['dynamic_boxes']  # (B, 3, T, 4)
        
        # Query 0: interacted object
        losses['object_box'] = self.box_loss(
            pred_dynamic[:, 0], gt_object_boxes
        ) * self.box_weight
        
        # Query 1: tool
        losses['tool_box'] = self.box_loss(
            pred_dynamic[:, 1], gt_tool_boxes
        ) * self.box_weight
        
        # Query 2: robot arm
        losses['robot_box'] = self.box_loss(
            pred_dynamic[:, 2], gt_robot_boxes
        ) * self.box_weight
        
        # === Static query losses ===
        pred_static = output['static_boxes']  # (B, 3, 4)
        
        # Query 0: start location
        losses['start_loc'] = self.box_loss(
            pred_static[:, 0], gt_start_location
        ) * self.box_weight
        
        # Query 1: end location
        losses['end_loc'] = self.box_loss(
            pred_static[:, 1], gt_end_location
        ) * self.box_weight
        
        # Query 2: target (optional)
        if gt_target is not None:
            losses['target'] = self.box_loss(
                pred_static[:, 2], gt_target
            ) * self.box_weight
        
        # === Confidence loss (optional) ===
        if gt_object_visible is not None:
            pred_conf = output['dynamic_confidence'][:, 0]  # (B, T)
            losses['conf'] = F.binary_cross_entropy(
                pred_conf, gt_object_visible.float()
            ) * self.conf_weight
        
        # === Temporal smoothness (boxes shouldn't jump) ===
        for i in range(3):
            pred_traj = pred_dynamic[:, i]  # (B, T, 4)
            temporal_diff = (pred_traj[:, 1:] - pred_traj[:, :-1]).abs().mean()
            losses[f'temporal_smooth_{i}'] = temporal_diff * self.temporal_weight
        
        # Total
        losses['total'] = sum(losses.values())
        
        return losses
    
    def box_loss(
        self, 
        pred: torch.Tensor,  # (..., 4)
        target: torch.Tensor,  # (..., 4)
    ) -> torch.Tensor:
        """L1 + GIoU loss for boxes."""
        l1 = F.l1_loss(pred, target)
        giou = self.giou_loss(pred, target)
        return l1 + giou * self.giou_weight
    
    def giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generalized IoU loss."""
        # Flatten to (N, 4)
        pred_flat = pred.reshape(-1, 4)
        target_flat = target.reshape(-1, 4)
        
        # IoU
        inter_x1 = torch.max(pred_flat[:, 0], target_flat[:, 0])
        inter_y1 = torch.max(pred_flat[:, 1], target_flat[:, 1])
        inter_x2 = torch.min(pred_flat[:, 2], target_flat[:, 2])
        inter_y2 = torch.min(pred_flat[:, 3], target_flat[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        pred_area = (pred_flat[:, 2] - pred_flat[:, 0]) * (pred_flat[:, 3] - pred_flat[:, 1])
        target_area = (target_flat[:, 2] - target_flat[:, 0]) * (target_flat[:, 3] - target_flat[:, 1])
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-8)
        
        # Enclosing box
        enc_x1 = torch.min(pred_flat[:, 0], target_flat[:, 0])
        enc_y1 = torch.min(pred_flat[:, 1], target_flat[:, 1])
        enc_x2 = torch.max(pred_flat[:, 2], target_flat[:, 2])
        enc_y2 = torch.max(pred_flat[:, 3], target_flat[:, 3])
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union_area) / (enc_area + 1e-8)
        
        return (1 - giou).mean()
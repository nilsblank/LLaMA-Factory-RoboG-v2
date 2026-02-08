"""
Training losses for the hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    def __init__(
        self,
        slot_box_weight: float = 2.0,
        query_box_weight: float = 2.0,
        slot_class_weight: float = 1.0,
        temporal_weight: float = 0.5,
        contrastive_weight: float = 1.0,
    ):
        super().__init__()
        self.slot_box_weight = slot_box_weight
        self.query_box_weight = query_box_weight
        self.slot_class_weight = slot_class_weight
        self.temporal_weight = temporal_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(
        self,
        output: Dict,
        gt_moved_object_boxes: torch.Tensor,    # (B, T, 4)
        gt_start_location: torch.Tensor,         # (B, 4)
        gt_end_location: torch.Tensor,           # (B, 4)
        gt_target_box: torch.Tensor = None,      # (B, 4) optional
        gt_robot_box: torch.Tensor = None,       # (B, T, 4) optional
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        B = gt_moved_object_boxes.shape[0]
        
        # === Slot box loss ===
        # Find which slot best matches the moved object
        slot_boxes = output['slot_boxes']  # (B, num_slots, T, 4)
        
        best_slot_loss = float('inf')
        for slot_idx in range(slot_boxes.shape[1]):
            slot_box = slot_boxes[:, slot_idx]  # (B, T, 4)
            loss = F.l1_loss(slot_box, gt_moved_object_boxes)
            if loss < best_slot_loss:
                best_slot_loss = loss
        
        losses['slot_box'] = best_slot_loss * self.slot_box_weight
        
        # === Query box losses ===
        losses['start_loc'] = F.l1_loss(
            output['query_boxes']['start_location'],
            gt_start_location
        ) * self.query_box_weight
        
        losses['end_loc'] = F.l1_loss(
            output['query_boxes']['end_location'],
            gt_end_location
        ) * self.query_box_weight
        
        if gt_target_box is not None:
            losses['target'] = F.l1_loss(
                output['query_boxes']['target_container'],
                gt_target_box
            ) * self.query_box_weight
        
        if gt_robot_box is not None:
            losses['robot'] = F.l1_loss(
                output['query_boxes']['robot_gripper'],
                gt_robot_box[:, gt_robot_box.shape[1] // 2]  # Middle frame
            ) * self.query_box_weight
        
        # === Temporal consistency (SlotContrast-style) ===
        slots_per_frame = output['slot_per_frame']  # (B, T, num_slots, H)
        slots_norm = F.normalize(slots_per_frame, dim=-1)
        
        s1 = slots_norm[:, :-1]  # (B, T-1, num_slots, H)
        s2 = slots_norm[:, 1:]   # (B, T-1, num_slots, H)
        
        sim = torch.einsum('btsh,btph->btsp', s1, s2)  # (B, T-1, num_slots, num_slots)
        
        # Target: diagonal should be 1 (same slot across frames)
        target = torch.eye(sim.shape[-1], device=sim.device)
        target = target.unsqueeze(0).unsqueeze(0).expand_as(sim)
        
        losses['temporal'] = F.mse_loss(sim, target) * self.temporal_weight
        
        # Total
        losses['total'] = sum(losses.values())
        
        return losses
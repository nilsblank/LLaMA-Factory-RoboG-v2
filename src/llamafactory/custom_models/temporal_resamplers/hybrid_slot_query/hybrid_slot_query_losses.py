"""
Training losses for the hybrid model.
"""

from typing import Dict, Optional
import torch
import torch.distributed as dist
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
        contrastive_temperature: float = 0.1,
        contrastive_queue_size: int = 2,
    ):
        super().__init__()
        self.slot_box_weight = slot_box_weight
        self.query_box_weight = query_box_weight
        self.slot_class_weight = slot_class_weight
        self.temporal_weight = temporal_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_queue_size = max(0, int(contrastive_queue_size))
        self.register_buffer("_contrastive_queue", torch.empty(0), persistent=False)
        self.use_constrastive_loss = True
        self.use_batch_contrastive = True

    def _push_queue(self, candidates: torch.Tensor) -> None:
        if self.contrastive_queue_size <= 0:
            return

        cand = candidates.detach().reshape(-1, candidates.shape[-1])
        if cand.numel() == 0:
            return

        if self._contrastive_queue.numel() == 0:
            queue = cand[-self.contrastive_queue_size :]
        else:
            queue = torch.cat([self._contrastive_queue.to(device=cand.device, dtype=cand.dtype), cand], dim=0)
            queue = queue[-self.contrastive_queue_size :]

        self._contrastive_queue = queue

    def _info_nce_exclusive_positive(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """InfoNCE with positives removed from denominator.

        Args:
            anchors: (G, N, D)
            positives: (G, N, D), positive for anchors[g, i] is positives[g, i]
            negatives: (G, M, D) optional extra negatives appended to denominator only
        """
        if anchors.numel() == 0:
            return anchors.sum() * 0.0

        logits_pos_set = torch.einsum("gnd,gmd->gnm", anchors, positives) / self.contrastive_temperature
        group_size = logits_pos_set.shape[-1]

        if group_size <= 1 and (negatives is None or negatives.shape[-2] == 0):
            return anchors.sum() * 0.0

        eye = torch.eye(group_size, device=anchors.device, dtype=torch.bool).unsqueeze(0)
        positive_logits = logits_pos_set.masked_select(eye).view(logits_pos_set.shape[0], group_size)

        neg_from_pos_set = logits_pos_set.masked_fill(eye, float("-inf"))

        if negatives is not None and negatives.shape[-2] > 0:
            logits_extra_neg = torch.einsum("gnd,gmd->gnm", anchors, negatives) / self.contrastive_temperature
            neg_logits = torch.cat([neg_from_pos_set, logits_extra_neg], dim=-1)
        else:
            neg_logits = neg_from_pos_set

        log_denom = torch.logsumexp(neg_logits, dim=-1)
        return -(positive_logits - log_denom).mean()

    def _slot_slot_contrastive_losses(self, slots_per_frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute SlotContrast-style losses from slots of shape (B, T, K, D)."""
        slots_norm = F.normalize(slots_per_frame, dim=-1, p=2.0)

        if self.use_batch_contrastive:

            slots = slots_norm.split(1) # [1xTxKxD]
            slots =  torch.cat(slots, dim=-2)  # 1xTxK*BxD 

        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.contrastive_temperature
        B,T,S,D = ss.shape
        ss = ss.reshape(B * T, S, S)        
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        batch_loss = F.cross_entropy(ss,target)
        return batch_loss

        # Adjacent frame pairs.
        # anchors/positives are (T-1, B, K, D), grouped by time index.
        anchors_t = slots_norm[:, :-1].permute(1, 0, 2, 3)
        positives_t = slots_norm[:, 1:].permute(1, 0, 2, 3)

        time_steps, batch_size, num_slots, dim = anchors_t.shape

        # Intra-video loss: for each (t, b), contrast among K slots only.
        intra_anchors = anchors_t.reshape(time_steps * batch_size, num_slots, dim)
        intra_pos = positives_t.reshape(time_steps * batch_size, num_slots, dim)
        intra_loss = self._info_nce_exclusive_positive(intra_anchors, intra_pos)

        # Batch-video loss: for each t, contrast across all B*K slots.
        batch_anchors = anchors_t.reshape(time_steps, batch_size * num_slots, dim)
        batch_pos = positives_t.reshape(time_steps, batch_size * num_slots, dim)

        dist_neg = None
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            local_count = torch.tensor([batch_pos.shape[1]], device=batch_pos.device, dtype=torch.long)
            gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(gathered_counts, local_count)
            counts = [int(c.item()) for c in gathered_counts]
            max_count = max(counts)

            if max_count > 0:
                if batch_pos.shape[1] < max_count:
                    pad_len = max_count - batch_pos.shape[1]
                    pad = torch.zeros(
                        (batch_pos.shape[0], pad_len, batch_pos.shape[2]),
                        device=batch_pos.device,
                        dtype=batch_pos.dtype,
                    )
                    batch_pos_padded = torch.cat([batch_pos.detach(), pad], dim=1)
                else:
                    batch_pos_padded = batch_pos.detach()

                gathered = [torch.zeros_like(batch_pos_padded) for _ in range(world_size)]
                dist.all_gather(gathered, batch_pos_padded)

                others = []
                for r, g in enumerate(gathered):
                    if r == rank:
                        continue
                    if counts[r] > 0:
                        others.append(g[:, : counts[r], :])

                if len(others) > 0:
                    dist_neg = torch.cat(others, dim=1)

        queue = None
        if self.contrastive_queue_size > 0 and self._contrastive_queue.numel() > 0:
            queue = self._contrastive_queue.to(device=batch_anchors.device, dtype=batch_anchors.dtype)
            queue = queue.unsqueeze(0).expand(time_steps, -1, -1)

        negatives = None
        if dist_neg is not None and queue is not None:
            negatives = torch.cat([dist_neg, queue], dim=1)
        elif dist_neg is not None:
            negatives = dist_neg
        elif queue is not None:
            negatives = queue

        batch_loss = self._info_nce_exclusive_positive(batch_anchors, batch_pos, negatives=negatives)

        queue_src = batch_pos.detach()
        if dist_neg is not None:
            queue_src = torch.cat([queue_src, dist_neg], dim=1)
        self._push_queue(queue_src)

        return {"intra": intra_loss, "batch": batch_loss}
        
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

        if self.use_constrastive_loss:
            slots_per_frame = output['slot_per_frame']  # (B, T, num_slots, H)
            contrastive_losses = self._slot_slot_contrastive_losses(slots_per_frame)
            losses['contrastive'] = contrastive_losses

        

        if all(x is not None for x in [gt_moved_object_boxes, gt_start_location, gt_end_location]):
                #skip object centric losses
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
        
        # === Slot contrastive losses (SlotContrast-style) ===
        
        # Total
        losses['total'] = sum(losses.values())
        
        return losses
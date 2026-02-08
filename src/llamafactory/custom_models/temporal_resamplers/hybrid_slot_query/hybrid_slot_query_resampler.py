"""
Hybrid Architecture:
- Slots for DISCOVERY (what moved)
- Queries for RELATIONS (where from, where to, what target)

This is more principled than pure slots or pure queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class HybridSlotQueryModel(nn.Module):
    """
    Stage 1: Slot Attention discovers moving objects (few slots, motion-biased)
    Stage 2: Typed Queries find relations/locations conditioned on discovered slots
    Stage 3: LLM reasons about everything
    """
    
    def __init__(
        self,
        vision_dim: int = 1280,
        hidden_dim: int = 512,
        num_slots: int = 3,          # Small! Robot + 1-2 objects
        num_frames: int = 16,
        num_heads: int = 8,
        qwen_model: str = "Qwen/Qwen2-0.5B",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.num_frames = num_frames
        
        # ============================================
        # STAGE 1: Slot-based Discovery
        # ============================================
        self.slot_discovery = MotionAwareSlotAttention(
            input_dim=vision_dim,
            slot_dim=hidden_dim,
            num_slots=num_slots,
            num_iterations=3,
        )
        
        # Slot classification: is this slot robot, moved_object, or background?
        self.slot_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # robot, object, background
        )
        
        # ============================================
        # STAGE 2: Query-based Relations
        # ============================================
        # Fixed, typed queries (not learned from scratch - semantically meaningful)
        self.relation_queries = nn.ModuleDict({
            'start_location': LearnableQuery(hidden_dim),
            'end_location': LearnableQuery(hidden_dim),
            'target_container': LearnableQuery(hidden_dim),
            'robot_gripper': LearnableQuery(hidden_dim),
        })
        
        self.relation_encoder = RelationQueryEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
        )
        
        # ============================================
        # STAGE 3: LLM Integration
        # ============================================
        from transformers import Qwen2ForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model)
        self.llm = Qwen2ForCausalLM.from_pretrained(qwen_model, torch_dtype=torch.bfloat16)
        
        llm_dim = self.llm.config.hidden_size
        
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Projections to LLM space
        self.slot_to_llm = nn.Linear(hidden_dim, llm_dim)
        self.query_to_llm = nn.Linear(hidden_dim, llm_dim)
        
        # ============================================
        # Output Heads
        # ============================================
        self.slot_box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        
        self.query_box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        
        # Temporal bounds for slots (when did this object move?)
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # start, end
            nn.Sigmoid(),
        )
        
        self.setup_special_tokens()
        
    def setup_special_tokens(self):
        special_tokens = ['<obj>', '</obj>', '<loc>', '</loc>', '<box>', '</box>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, T, N, D)
    ) -> Dict[str, torch.Tensor]:
        B, T, N, D = visual_features.shape
        
        # ============================================
        # STAGE 1: Discover what moved
        # ============================================
        slot_output = self.slot_discovery(visual_features)
        
        slot_features = slot_output['slots']              # (B, num_slots, H)
        slot_per_frame = slot_output['slots_per_frame']   # (B, T, num_slots, H)
        slot_attn = slot_output['attention']              # (B, T, num_slots, N)
        
        # Classify slots
        slot_classes = self.slot_classifier(slot_features)  # (B, num_slots, 3)
        
        # Predict boxes per frame
        slot_boxes = self.slot_box_head(slot_per_frame)     # (B, T, num_slots, 4)
        slot_boxes = slot_boxes.permute(0, 2, 1, 3)         # (B, num_slots, T, 4)
        
        # Predict temporal bounds
        temporal_bounds = self.temporal_head(slot_features)  # (B, num_slots, 2)
        
        # ============================================
        # STAGE 2: Find relations conditioned on slots
        # ============================================
        
        # Get the "moved object" slot (highest motion / classified as object)
        object_probs = slot_classes.softmax(dim=-1)[:, :, 1]  # (B, num_slots) - prob of being object
        moved_object_idx = object_probs.argmax(dim=-1)        # (B,)
        
        # Extract moved object features
        moved_object_features = torch.stack([
            slot_features[b, moved_object_idx[b]] for b in range(B)
        ])  # (B, H)
        
        # Get relation query features
        query_features = {}
        query_boxes = {}
        
        for name, query_module in self.relation_queries.items():
            # Query embedding
            q = query_module(B, visual_features.device)  # (B, 1, H)
            
            # Encode with cross-attention to visual + slot conditioning
            encoded = self.relation_encoder(
                query=q,
                visual_features=visual_features,
                slot_features=slot_features,
                moved_object_features=moved_object_features,
            )  # (B, 1, H)
            
            query_features[name] = encoded.squeeze(1)  # (B, H)
            query_boxes[name] = self.query_box_head(encoded).squeeze(1)  # (B, 4)
        
        return {
            # Slot outputs (discovered objects)
            'slot_features': slot_features,
            'slot_per_frame': slot_per_frame,
            'slot_boxes': slot_boxes,
            'slot_classes': slot_classes,
            'slot_attention': slot_attn,
            'temporal_bounds': temporal_bounds,
            'moved_object_idx': moved_object_idx,
            
            # Query outputs (relations)
            'query_features': query_features,
            'query_boxes': query_boxes,
        }
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        prompt: str = "Describe what happened. What object moved, from where, to where?",
        max_new_tokens: int = 150,
    ) -> Dict:
        B = visual_features.shape[0]
        assert B == 1, "Generation supports batch size 1"
        
        device = visual_features.device
        
        # Forward
        output = self.forward(visual_features)
        
        # Build context
        context = self.build_context(output)
        
        # Project features to LLM space
        slot_tokens = self.slot_to_llm(output['slot_features'])  # (1, num_slots, llm_dim)
        
        query_tokens = torch.stack([
            self.query_to_llm(output['query_features'][name])
            for name in self.relation_queries.keys()
        ], dim=1)  # (1, num_queries, llm_dim)
        
        # Concatenate all visual tokens
        visual_tokens = torch.cat([slot_tokens, query_tokens], dim=1)  # (1, num_slots + num_queries, llm_dim)
        
        # Tokenize prompt
        full_prompt = f"{context}\n\nQuestion: {prompt}\n\nAnswer:"
        text_inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)
        text_embeds = self.llm.model.embed_tokens(text_inputs.input_ids)
        
        # Combine
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("Answer:")[-1].strip()
        
        return {
            'text': answer,
            'moved_object_box': output['slot_boxes'][0, output['moved_object_idx'][0]].cpu(),
            'start_location_box': output['query_boxes']['start_location'][0].cpu(),
            'end_location_box': output['query_boxes']['end_location'][0].cpu(),
            'target_box': output['query_boxes']['target_container'][0].cpu(),
            'robot_box': output['query_boxes']['robot_gripper'][0].cpu(),
            'temporal_bounds': output['temporal_bounds'][0, output['moved_object_idx'][0]].cpu(),
        }
    
    def build_context(self, output: Dict) -> str:
        """Build text context from discovered slots and queries."""
        
        lines = ["Discovered entities:"]
        
        # Slots
        slot_classes = output['slot_classes'][0].softmax(dim=-1)  # (num_slots, 3)
        slot_boxes = output['slot_boxes'][0]  # (num_slots, T, 4)
        temporal = output['temporal_bounds'][0]  # (num_slots, 2)
        
        class_names = ['robot', 'moved_object', 'background']
        
        for i in range(self.num_slots):
            cls_idx = slot_classes[i].argmax().item()
            cls_name = class_names[cls_idx]
            conf = slot_classes[i, cls_idx].item()
            
            if cls_name == 'background' or conf < 0.3:
                continue
            
            mid_box = slot_boxes[i, self.num_frames // 2]
            start_t = int(temporal[i, 0].item() * self.num_frames)
            end_t = int(temporal[i, 1].item() * self.num_frames)
            
            lines.append(
                f"  <obj>{cls_name}</obj> "
                f"<box>[{mid_box[0]:.2f},{mid_box[1]:.2f},{mid_box[2]:.2f},{mid_box[3]:.2f}]</box> "
                f"(frames {start_t}-{end_t})"
            )
        
        # Queries (relations)
        lines.append("\nSpatial relations:")
        
        query_names = {
            'start_location': 'started at',
            'end_location': 'ended at', 
            'target_container': 'target container',
            'robot_gripper': 'robot gripper at',
        }
        
        for name, box in output['query_boxes'].items():
            b = box[0]
            readable = query_names.get(name, name)
            lines.append(
                f"  <loc>{readable}</loc> "
                f"<box>[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}]</box>"
            )
        
        return "\n".join(lines)


class MotionAwareSlotAttention(nn.Module):
    """
    Slot attention biased toward moving regions.
    
    Key differences from vanilla slot attention:
    1. Motion features added to input
    2. Temporal consistency (same slot = same object across frames)
    3. Few slots (you said 1-2 objects + robot)
    """
    
    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_slots: int = 3,
        num_iterations: int = 3,
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.scale = slot_dim ** -0.5
        
        # Slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slots_sigma = nn.Parameter(torch.ones(1, num_slots, slot_dim) * 0.1)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.input_norm = nn.LayerNorm(slot_dim)
        
        # Motion encoding
        self.motion_proj = nn.Sequential(
            nn.Linear(input_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        
        # Slot attention
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        
        self.slot_norm = nn.LayerNorm(slot_dim)
        
        # Temporal consistency
        self.temporal_attn = nn.MultiheadAttention(slot_dim, 8, batch_first=True)
        self.temporal_norm = nn.LayerNorm(slot_dim)
        
    def compute_motion(self, features: torch.Tensor) -> torch.Tensor:
        """Compute motion signal between consecutive frames."""
        # features: (B, T, N, D)
        motion = features[:, 1:] - features[:, :-1]  # (B, T-1, N, D)
        motion = F.pad(motion, (0, 0, 0, 0, 0, 1))   # (B, T, N, D)
        return motion
        
    def forward(self, visual_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, N, D = visual_features.shape
        device = visual_features.device
        
        # Project input
        features = self.input_norm(self.input_proj(visual_features))  # (B, T, N, H)
        
        # Add motion bias
        motion = self.compute_motion(visual_features)
        motion_features = self.motion_proj(motion)  # (B, T, N, H)
        features = features + 0.5 * motion_features  # Weighted motion bias
        
        # Initialize slots
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            B, self.num_slots, self.slot_dim, device=device
        )
        
        all_slots = []
        all_attn = []
        
        for t in range(T):
            frame_features = features[:, t]  # (B, N, H)
            
            # Iterative slot attention
            frame_slots = slots
            for _ in range(self.num_iterations):
                frame_slots_norm = self.slot_norm(frame_slots)
                
                q = self.to_q(frame_slots_norm)
                k = self.to_k(frame_features)
                v = self.to_v(frame_features)
                
                # Attention with competition
                attn_logits = torch.einsum('bsh,bnh->bsn', q, k) * self.scale
                attn = F.softmax(attn_logits, dim=1)  # Competition over slots
                attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
                
                updates = torch.einsum('bsn,bnh->bsh', attn_norm, v)
                
                frame_slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    frame_slots.reshape(-1, self.slot_dim),
                ).reshape(B, self.num_slots, -1)
                
                frame_slots = frame_slots + self.mlp(frame_slots)
            
            all_slots.append(frame_slots)
            all_attn.append(attn)
            
            # Use for next frame (temporal consistency)
            slots = frame_slots
        
        slots_per_frame = torch.stack(all_slots, dim=1)  # (B, T, num_slots, H)
        attention = torch.stack(all_attn, dim=1)          # (B, T, num_slots, N)
        
        # Temporal aggregation
        slots_flat = slots_per_frame.permute(0, 2, 1, 3).reshape(B * self.num_slots, T, -1)
        slots_temporal, _ = self.temporal_attn(slots_flat, slots_flat, slots_flat)
        slots_temporal = self.temporal_norm(slots_temporal + slots_flat)
        
        # Mean pool for final slot representation
        slots_agg = slots_temporal.mean(dim=1).reshape(B, self.num_slots, -1)
        
        return {
            'slots': slots_agg,              # (B, num_slots, H)
            'slots_per_frame': slots_per_frame,  # (B, T, num_slots, H)
            'attention': attention,          # (B, T, num_slots, N)
        }


class LearnableQuery(nn.Module):
    """A single learnable query with semantic initialization."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.query.expand(batch_size, -1, -1).to(device)


class RelationQueryEncoder(nn.Module):
    """
    Encodes relation queries by attending to:
    1. Visual features (to find locations)
    2. Discovered slots (to relate to objects)
    3. Moved object specifically (main subject of relations)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            RelationQueryLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        query: torch.Tensor,               # (B, 1, H)
        visual_features: torch.Tensor,     # (B, T, N, D)
        slot_features: torch.Tensor,       # (B, num_slots, H)
        moved_object_features: torch.Tensor,  # (B, H)
    ) -> torch.Tensor:
        B, T, N, D = visual_features.shape
        
        # Flatten visual features
        visual_flat = visual_features.mean(dim=1)  # (B, N, D) - temporal average
        
        # Add moved object as context
        moved_obj = moved_object_features.unsqueeze(1)  # (B, 1, H)
        context = torch.cat([slot_features, moved_obj], dim=1)  # (B, num_slots+1, H)
        
        # Process query
        for layer in self.layers:
            query = layer(query, visual_flat, context)
        
        return query


class RelationQueryLayer(nn.Module):
    """Single layer for relation query processing."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        # Cross-attention to visual features
        self.visual_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to slot context
        self.context_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Visual projection (in case dims don't match)
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        query: torch.Tensor,    # (B, 1, H)
        visual: torch.Tensor,   # (B, N, D)
        context: torch.Tensor,  # (B, num_slots+1, H)
    ) -> torch.Tensor:
        
        # Project visual if needed
        visual = self.visual_proj(visual)
        
        # Cross-attention to visual
        q = self.visual_norm(query)
        attn_out, _ = self.visual_attn(q, visual, visual)
        query = query + attn_out
        
        # Cross-attention to slot context
        q = self.context_norm(query)
        attn_out, _ = self.context_attn(q, context, context)
        query = query + attn_out
        
        # FFN
        query = query + self.ffn(self.ffn_norm(query))
        
        return query
"""
Text-Conditioned Queries with Temporal Extension.

Key insight: You KNOW the semantic categories (interacted_object, tool, robot).
You just need to find WHERE they are in each frame and TRACK them.

This is simpler and more stable than slot attention for your use case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class TextConditionedTemporalQueries(nn.Module):
    """
    Text-conditioned queries that track across frames.
    
    Query types:
    - DYNAMIC: Track across frames (interacted_object, tool, robot_arm)
    - STATIC: Single box, don't change much (start_location, end_location, target)
    """
    
    def __init__(
        self,
        vision_dim: int = 1280,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_frames: int = 16,
        text_encoder: str = "Qwen/Qwen2-0.5B",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        
        # === Text Encoder (frozen) ===
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.text_model = AutoModel.from_pretrained(text_encoder)
        self.text_dim = self.text_model.config.hidden_size
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Project text to query space
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # === Visual Projection ===
        self.vis_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # === Temporal Encoding ===
        self.register_buffer(
            'time_encoding', 
            self._create_sinusoidal(num_frames, hidden_dim)
        )
        
        # === Query Types ===
        # Dynamic queries: tracked per-frame
        self.dynamic_query_texts = [
            "the object being manipulated",
            "robot gripper or tool", 
            "robot arm",
        ]
        
        # Static queries: single location
        self.static_query_texts = [
            "location where object started",
            "location where object ended",
            "target container or destination",
        ]
        
        # === Query Processing ===
        # For dynamic queries: cross-attention + temporal
        self.dynamic_encoder = DynamicQueryEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        # For static queries: cross-attention only (no tracking needed)
        self.static_encoder = StaticQueryEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers // 2,
        )
        
        # === Output Heads ===
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Cache encoded query texts
        self._cached_queries = None
        
    def _create_sinusoidal(self, length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
    
    @torch.no_grad()
    def encode_query_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode query texts using frozen text encoder."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to(device)
        
        outputs = self.text_model(**tokens)
        
        # Mean pooling over sequence
        mask = tokens.attention_mask.unsqueeze(-1)
        text_features = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        
        return text_features  # (num_queries, text_dim)
    
    def get_query_embeddings(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get query embeddings, using cache if available."""
        
        if self._cached_queries is None or self._cached_queries[0].device != device:
            # Encode dynamic queries
            dynamic_text_feats = self.encode_query_texts(self.dynamic_query_texts, device)
            dynamic_queries = self.text_proj(dynamic_text_feats)  # (num_dynamic, H)
            
            # Encode static queries
            static_text_feats = self.encode_query_texts(self.static_query_texts, device)
            static_queries = self.text_proj(static_text_feats)  # (num_static, H)
            
            self._cached_queries = (dynamic_queries, static_queries)
        
        dynamic_q, static_q = self._cached_queries
        
        # Expand for batch
        dynamic_q = dynamic_q.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_dynamic, H)
        static_q = static_q.unsqueeze(0).expand(batch_size, -1, -1)    # (B, num_static, H)
        
        return dynamic_q, static_q
    
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, T, N, D)
    ) -> Dict[str, torch.Tensor]:
        B, T, N, D = visual_features.shape
        device = visual_features.device
        
        # === Project visual features ===
        vis = self.vis_proj(visual_features)  # (B, T, N, H)
        
        # Add temporal encoding
        time_enc = self.time_encoding[:T].view(1, T, 1, -1)  # (1, T, 1, H)
        vis = vis + time_enc
        
        # === Get query embeddings ===
        dynamic_queries, static_queries = self.get_query_embeddings(B, device)
        
        num_dynamic = len(self.dynamic_query_texts)
        num_static = len(self.static_query_texts)
        
        # === Process dynamic queries (per-frame tracking) ===
        dynamic_output = self.dynamic_encoder(
            queries=dynamic_queries,      # (B, num_dynamic, H)
            visual_features=vis,          # (B, T, N, H)
            time_encoding=self.time_encoding[:T],
        )
        # dynamic_features: (B, num_dynamic, T, H)
        # Contains tracked features per frame
        
        dynamic_features = dynamic_output['features']
        
        # === Process static queries (single location) ===
        # Use temporal average or key frames
        vis_static = torch.cat([
            vis[:, 0],           # First frame
            vis[:, T // 2],      # Middle frame  
            vis[:, -1],          # Last frame
        ], dim=1)  # (B, 3*N, H)
        
        static_output = self.static_encoder(
            queries=static_queries,   # (B, num_static, H)
            visual_features=vis_static,
        )
        # static_features: (B, num_static, H)
        
        static_features = static_output['features']
        
        # === Predict boxes ===
        
        # Dynamic: boxes per frame
        dynamic_boxes = self.box_head(dynamic_features)  # (B, num_dynamic, T, 4)
        dynamic_conf = self.confidence_head(dynamic_features).squeeze(-1)  # (B, num_dynamic, T)
        
        # Static: single box per query
        static_boxes = self.box_head(static_features)  # (B, num_static, 4)
        static_conf = self.confidence_head(static_features).squeeze(-1)  # (B, num_static)
        
        return {
            # Dynamic query outputs (tracked)
            'dynamic_features': dynamic_features,    # (B, num_dynamic, T, H)
            'dynamic_boxes': dynamic_boxes,          # (B, num_dynamic, T, 4)
            'dynamic_confidence': dynamic_conf,      # (B, num_dynamic, T)
            'dynamic_names': self.dynamic_query_texts,
            
            # Static query outputs
            'static_features': static_features,      # (B, num_static, H)
            'static_boxes': static_boxes,            # (B, num_static, 4)
            'static_confidence': static_conf,        # (B, num_static)
            'static_names': self.static_query_texts,
            
            # For LLM
            'all_features': torch.cat([
                dynamic_features.mean(dim=2),  # Aggregate dynamic over time: (B, num_dynamic, H)
                static_features,                # (B, num_static, H)
            ], dim=1),  # (B, num_dynamic + num_static, H)
        }


class DynamicQueryEncoder(nn.Module):
    """
    Encodes dynamic queries with per-frame tracking.
    
    Key: Queries are propagated through time, attending to each frame
    while maintaining temporal consistency.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Per-frame cross-attention
        self.frame_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Temporal self-attention (queries attend across time)
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers // 2)
        ])
        
    def forward(
        self,
        queries: torch.Tensor,         # (B, num_queries, H)
        visual_features: torch.Tensor,  # (B, T, N, H)
        time_encoding: torch.Tensor,    # (T, H)
    ) -> Dict[str, torch.Tensor]:
        B, T, N, H = visual_features.shape
        num_queries = queries.shape[1]
        
        # Process each frame, propagating queries through time
        all_frame_queries = []
        
        current_queries = queries  # (B, num_queries, H)
        
        for t in range(T):
            frame_vis = visual_features[:, t]  # (B, N, H)
            
            # Add time encoding to queries (so they know which frame)
            time_bias = time_encoding[t].unsqueeze(0).unsqueeze(0)  # (1, 1, H)
            frame_queries = current_queries + time_bias
            
            # Cross-attention to this frame's visual features
            for layer in self.frame_layers:
                frame_queries = layer(frame_queries, frame_vis)
            
            all_frame_queries.append(frame_queries)
            
            # Propagate to next frame (with residual)
            current_queries = current_queries + 0.1 * (frame_queries - current_queries)
        
        # Stack: (B, T, num_queries, H) -> (B, num_queries, T, H)
        queries_per_frame = torch.stack(all_frame_queries, dim=1).permute(0, 2, 1, 3)
        
        # Temporal self-attention: queries attend across their own time dimension
        # Reshape: (B, num_queries, T, H) -> (B * num_queries, T, H)
        queries_flat = queries_per_frame.reshape(B * num_queries, T, H)
        
        for layer in self.temporal_layers:
            queries_flat = layer(queries_flat)
        
        # Reshape back: (B * num_queries, T, H) -> (B, num_queries, T, H)
        queries_per_frame = queries_flat.reshape(B, num_queries, T, H)
        
        return {
            'features': queries_per_frame,  # (B, num_queries, T, H)
        }


class StaticQueryEncoder(nn.Module):
    """
    Encodes static queries (locations that don't move).
    Simpler than dynamic - just cross-attention, no temporal tracking.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        queries: torch.Tensor,         # (B, num_queries, H)
        visual_features: torch.Tensor,  # (B, M, H) - M = concatenated key frames
    ) -> Dict[str, torch.Tensor]:
        
        for layer in self.layers:
            queries = layer(queries, visual_features)
        
        return {
            'features': queries,  # (B, num_queries, H)
        }


class CrossAttentionLayer(nn.Module):
    """Standard cross-attention layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        q = self.norm1(queries)
        attn_out, _ = self.cross_attn(q, kv, kv)
        queries = queries + attn_out
        
        # FFN
        queries = queries + self.ffn(self.norm2(queries))
        
        return queries


class TemporalAttentionLayer(nn.Module):
    """Self-attention across time dimension."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        
        # Self-attention across time
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
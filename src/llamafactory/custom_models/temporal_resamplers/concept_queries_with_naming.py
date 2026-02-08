"""
Concept-Level Queries + Open Vocabulary Naming

Stage 1: Concept queries find semantic regions
         "interacted object" → finds the thing being manipulated
         
Stage 2: LLM names what was found
         [visual features of found region] → "red cube"
         
This is like TimeChat but with semantic concepts instead of frame numbers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class ConceptQueryWithNaming(nn.Module):
    """
    Concept-level queries (like TimeChat's frame queries, but semantic).
    Plus explicit naming via LLM.
    
    Query texts are CONCEPTS, not object names:
    - "the object being manipulated" (not "red cube")
    - "the robot's gripper" (not "Franka gripper")
    - "where the object started" (not "table corner")
    
    The LLM then NAMES what each query found.
    """
    
    def __init__(
        self,
        vision_dim: int = 1280,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_frames: int = 16,
        qwen_model: str = "Qwen/Qwen2-0.5B",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        
        # === Concept Queries (semantic roles, not object names) ===
        self.concept_queries = {
            # Dynamic (tracked across frames)
            'dynamic': [
                "the object being manipulated by the robot",
                "the robot gripper or end effector tool",
                "the robot arm",
            ],
            # Static (single location)
            'static': [
                "the location where the object started",
                "the location where the object was placed",
                "the target container or destination",
            ],
        }
        
        # === Text Encoder (shared with LLM) ===
        from transformers import AutoTokenizer, AutoModel, Qwen2ForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model)
        
        # For encoding concept queries
        self.text_encoder = AutoModel.from_pretrained(qwen_model)
        self.text_dim = self.text_encoder.config.hidden_size
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # For naming + reasoning
        self.llm = Qwen2ForCausalLM.from_pretrained(qwen_model, torch_dtype=torch.bfloat16)
        
        for param in self.llm.parameters():
            param.requires_grad = False
        
        llm_dim = self.llm.config.hidden_size
        
        # === Projections ===
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.vis_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.to_llm = nn.Linear(hidden_dim, llm_dim)
        
        # === Query Encoders ===
        self.dynamic_encoder = DynamicConceptEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        self.static_encoder = StaticConceptEncoder(
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
        
        # === Temporal ===
        self.register_buffer('time_enc', self._sinusoidal(num_frames, hidden_dim))
        
        # === Naming Head (extracts visual features for LLM to name) ===
        self.naming_proj = nn.Sequential(
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        
        # Cache
        self._query_cache = None
        
        # Setup special tokens
        self.setup_tokens()
        
    def _sinusoidal(self, length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
    
    def setup_tokens(self):
        special = [
            '<obj>', '</obj>',      # Object mention  
            '<loc>', '</loc>',      # Location mention
            '<box>', '</box>',      # Bounding box
            '<name>', '</name>',    # Named entity
        ]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special})
        self.llm.resize_token_embeddings(len(self.tokenizer))
    
    @torch.no_grad()
    def encode_concepts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode concept query texts."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to(device)
        
        outputs = self.text_encoder(**tokens)
        
        # Mean pool
        mask = tokens.attention_mask.unsqueeze(-1)
        features = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        
        return features
    
    def get_concept_queries(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concept query embeddings."""
        if self._query_cache is None or self._query_cache[0].device != device:
            dynamic_feats = self.encode_concepts(self.concept_queries['dynamic'], device)
            static_feats = self.encode_concepts(self.concept_queries['static'], device)
            
            dynamic_q = self.text_proj(dynamic_feats)
            static_q = self.text_proj(static_feats)
            
            self._query_cache = (dynamic_q, static_q)
        
        dynamic_q, static_q = self._query_cache
        
        return (
            dynamic_q.unsqueeze(0).expand(batch_size, -1, -1),
            static_q.unsqueeze(0).expand(batch_size, -1, -1),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, T, N, D)
    ) -> Dict[str, torch.Tensor]:
        B, T, N, D = visual_features.shape
        device = visual_features.device
        
        # Project visual
        vis = self.vis_proj(visual_features)
        vis = vis + self.time_enc[:T].view(1, T, 1, -1)
        
        # Get concept queries
        dynamic_q, static_q = self.get_concept_queries(B, device)
        
        # Process dynamic queries (tracked)
        dynamic_out = self.dynamic_encoder(dynamic_q, vis, self.time_enc[:T])
        dynamic_features = dynamic_out['features']  # (B, num_dynamic, T, H)
        dynamic_attention = dynamic_out['attention']  # (B, num_dynamic, T, N)
        
        # Process static queries
        vis_keyframes = torch.cat([vis[:, 0], vis[:, T//2], vis[:, -1]], dim=1)
        static_out = self.static_encoder(static_q, vis_keyframes)
        static_features = static_out['features']  # (B, num_static, H)
        static_attention = static_out['attention']  # (B, num_static, 3*N)
        
        # Predict boxes
        dynamic_boxes = self.box_head(dynamic_features)  # (B, num_dynamic, T, 4)
        dynamic_conf = self.confidence_head(dynamic_features).squeeze(-1)
        
        static_boxes = self.box_head(static_features)  # (B, num_static, 4)
        static_conf = self.confidence_head(static_features).squeeze(-1)
        
        # Prepare features for naming (will be used by LLM)
        naming_features = {
            'dynamic': self.naming_proj(dynamic_features.mean(dim=2)),  # (B, num_dynamic, llm_dim)
            'static': self.naming_proj(static_features),  # (B, num_static, llm_dim)
        }
        
        return {
            'dynamic_features': dynamic_features,
            'dynamic_boxes': dynamic_boxes,
            'dynamic_confidence': dynamic_conf,
            'dynamic_attention': dynamic_attention,
            
            'static_features': static_features,
            'static_boxes': static_boxes,
            'static_confidence': static_conf,
            'static_attention': static_attention,
            
            'naming_features': naming_features,
            
            'concept_names': {
                'dynamic': self.concept_queries['dynamic'],
                'static': self.concept_queries['static'],
            },
        }
    
    @torch.no_grad()
    def name_object(
        self,
        naming_features: torch.Tensor,  # (1, llm_dim)
        concept: str,
    ) -> str:
        """Use LLM to name what a query found."""
        device = naming_features.device
        
        prompt = f"Based on the visual features, what specific object is '{concept}'? Answer with just the object name (e.g., 'red cube', 'metal bowl', 'wooden block').\n\nObject name:"
        
        tokens = self.tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = self.llm.model.embed_tokens(tokens.input_ids)
        
        # Prepend visual features
        inputs_embeds = torch.cat([naming_features.unsqueeze(1), text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        name = generated.split("Object name:")[-1].strip().split('\n')[0]
        
        return name
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        prompt: str = "Describe what the robot did, naming the specific objects involved.",
        max_new_tokens: int = 200,
    ) -> Dict:
        B = visual_features.shape[0]
        device = visual_features.device
        
        # Forward
        output = self.forward(visual_features)
        
        # === Name each discovered entity ===
        discovered_names = {}
        
        # Name dynamic entities
        for i, concept in enumerate(output['concept_names']['dynamic']):
            if output['dynamic_confidence'][0, i].mean() > 0.3:
                feat = output['naming_features']['dynamic'][0, i:i+1]
                name = self.name_object(feat, concept)
                discovered_names[concept] = name
            else:
                discovered_names[concept] = None
        
        # Name static entities  
        for i, concept in enumerate(output['concept_names']['static']):
            if output['static_confidence'][0, i] > 0.3:
                feat = output['naming_features']['static'][0, i:i+1]
                name = self.name_object(feat, concept)
                discovered_names[concept] = name
            else:
                discovered_names[concept] = None
        
        # === Build grounded context with names ===
        context = self.build_named_context(output, discovered_names)
        
        # === Generate description ===
        full_prompt = f"{context}\n\nQuestion: {prompt}\n\nAnswer:"
        
        # Project all features to LLM
        all_features = torch.cat([
            output['naming_features']['dynamic'].mean(dim=0, keepdim=True),  # (1, num_dynamic, llm_dim)
            output['naming_features']['static'],  # (1, num_static, llm_dim)
        ], dim=1)[0]  # (num_total, llm_dim)
        
        tokens = self.tokenizer(full_prompt, return_tensors="pt").to(device)
        text_embeds = self.llm.model.embed_tokens(tokens.input_ids)
        
        inputs_embeds = torch.cat([all_features.unsqueeze(0), text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated.split("Answer:")[-1].strip()
        
        return {
            'text': answer,
            'discovered_names': discovered_names,
            'boxes': {
                'interacted_object': output['dynamic_boxes'][0, 0].cpu(),
                'tool': output['dynamic_boxes'][0, 1].cpu(),
                'robot': output['dynamic_boxes'][0, 2].cpu(),
                'start_location': output['static_boxes'][0, 0].cpu(),
                'end_location': output['static_boxes'][0, 1].cpu(),
                'target': output['static_boxes'][0, 2].cpu(),
            },
        }
    
    def build_named_context(self, output: Dict, names: Dict) -> str:
        """Build context with discovered names."""
        lines = ["Discovered entities:"]
        
        # Dynamic
        dynamic_concepts = output['concept_names']['dynamic']
        for i, concept in enumerate(dynamic_concepts):
            conf = output['dynamic_confidence'][0, i].mean().item()
            if conf < 0.3:
                continue
                
            name = names.get(concept, "unknown")
            boxes = output['dynamic_boxes'][0, i]
            
            b_mid = boxes[self.num_frames // 2]
            
            if name:
                lines.append(
                    f"  <obj>{concept}</obj> = <name>{name}</name> "
                    f"<box>[{b_mid[0]:.2f},{b_mid[1]:.2f},{b_mid[2]:.2f},{b_mid[3]:.2f}]</box>"
                )
        
        # Static
        static_concepts = output['concept_names']['static']
        for i, concept in enumerate(static_concepts):
            conf = output['static_confidence'][0, i].item()
            if conf < 0.3:
                continue
                
            name = names.get(concept, "unknown")
            box = output['static_boxes'][0, i]
            
            if name:
                lines.append(
                    f"  <loc>{concept}</loc> = <name>{name}</name> "
                    f"<box>[{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}]</box>"
                )
        
        return "\n".join(lines)


class DynamicConceptEncoder(nn.Module):
    """Encodes dynamic concept queries with temporal tracking."""
    
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ConceptAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.temporal_layers = nn.ModuleList([
            TemporalSelfAttention(hidden_dim, num_heads)
            for _ in range(num_layers // 2)
        ])
        
    def forward(
        self,
        queries: torch.Tensor,  # (B, num_q, H)
        visual: torch.Tensor,   # (B, T, N, H)
        time_enc: torch.Tensor,  # (T, H)
    ) -> Dict[str, torch.Tensor]:
        B, T, N, H = visual.shape
        num_q = queries.shape[1]
        
        all_queries = []
        all_attention = []
        
        current = queries
        
        for t in range(T):
            frame_vis = visual[:, t]
            frame_q = current + time_enc[t].unsqueeze(0).unsqueeze(0)
            
            for layer in self.layers:
                frame_q, attn = layer(frame_q, frame_vis)
            
            all_queries.append(frame_q)
            all_attention.append(attn)
            
            # Propagate
            current = current + 0.1 * (frame_q - current)
        
        # (B, num_q, T, H)
        queries_per_frame = torch.stack(all_queries, dim=2)
        attention = torch.stack(all_attention, dim=2)  # (B, num_q, T, N)
        
        # Temporal self-attention
        q_flat = queries_per_frame.reshape(B * num_q, T, H)
        for layer in self.temporal_layers:
            q_flat = layer(q_flat)
        queries_per_frame = q_flat.reshape(B, num_q, T, H)
        
        return {
            'features': queries_per_frame,
            'attention': attention,
        }


class StaticConceptEncoder(nn.Module):
    """Encodes static concept queries (locations)."""
    
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ConceptAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        queries: torch.Tensor,  # (B, num_q, H)
        visual: torch.Tensor,   # (B, M, H) - keyframes concatenated
    ) -> Dict[str, torch.Tensor]:
        
        attn = None
        for layer in self.layers:
            queries, attn = layer(queries, visual)
        
        return {
            'features': queries,
            'attention': attn,
        }


class ConceptAttentionLayer(nn.Module):
    """Cross-attention layer that returns attention weights."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(
        self, 
        queries: torch.Tensor,  # (B, num_q, H)
        kv: torch.Tensor,       # (B, N, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, num_q, H = queries.shape
        N = kv.shape[1]
        
        # Cross-attention
        q = self.norm1(queries)
        
        q = self.q_proj(q).view(B, num_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)  # (B, heads, num_q, N)
        
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, num_q, H)
        attn_out = self.out_proj(attn_out)
        
        queries = queries + attn_out
        
        # FFN
        queries = queries + self.ffn(self.norm2(queries))
        
        # Return mean attention over heads for visualization
        attn_mean = attn_weights.mean(dim=1)  # (B, num_q, N)
        
        return queries, attn_mean


class TemporalSelfAttention(nn.Module):
    """Self-attention across time."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
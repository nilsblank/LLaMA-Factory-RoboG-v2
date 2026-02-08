"""
Full VLM with text-conditioned temporal queries.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class TextConditionedRobotVLM(nn.Module):
    """
    Complete model:
    1. Text-conditioned queries find objects/locations
    2. Dynamic queries track across frames
    3. Static queries find locations
    4. LLM generates grounded description
    """
    
    def __init__(
        self,
        vision_dim: int = 1280,
        hidden_dim: int = 512,
        num_frames: int = 16,
        qwen_model: str = "Qwen/Qwen2-0.5B",
    ):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Query module
        self.query_module = TextConditionedTemporalQueries(
            vision_dim=vision_dim,
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            text_encoder=qwen_model,  # Reuse Qwen for text encoding
        )
        
        # LLM
        from transformers import Qwen2ForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model)
        self.llm = Qwen2ForCausalLM.from_pretrained(qwen_model, torch_dtype=torch.bfloat16)
        
        llm_dim = self.llm.config.hidden_size
        
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Project query features to LLM
        self.to_llm = nn.Linear(hidden_dim, llm_dim)
        
        # Special tokens
        self.setup_tokens()
        
    def setup_tokens(self):
        special = ['<obj>', '</obj>', '<loc>', '</loc>', '<box>', '</box>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, visual_features: torch.Tensor) -> Dict:
        return self.query_module(visual_features)
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        prompt: str = "Describe what the robot did.",
        max_new_tokens: int = 150,
    ) -> Dict:
        B = visual_features.shape[0]
        device = visual_features.device
        
        # Get query outputs
        output = self.forward(visual_features)
        
        # Build grounded context
        context = self.build_context(output)
        
        # Project to LLM space
        query_features = output['all_features']  # (B, num_queries, H)
        llm_tokens = self.to_llm(query_features)  # (B, num_queries, llm_dim)
        
        # Tokenize prompt
        full_prompt = f"{context}\n\nQuestion: {prompt}\n\nAnswer:"
        text_inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)
        text_embeds = self.llm.model.embed_tokens(text_inputs.input_ids)
        
        # Combine
        inputs_embeds = torch.cat([llm_tokens, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        
        # Generate
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
            'interacted_object_boxes': output['dynamic_boxes'][0, 0].cpu(),  # (T, 4)
            'tool_boxes': output['dynamic_boxes'][0, 1].cpu(),               # (T, 4)
            'robot_boxes': output['dynamic_boxes'][0, 2].cpu(),              # (T, 4)
            'start_location': output['static_boxes'][0, 0].cpu(),            # (4,)
            'end_location': output['static_boxes'][0, 1].cpu(),              # (4,)
            'target': output['static_boxes'][0, 2].cpu(),                    # (4,)
        }
    
    def build_context(self, output: Dict) -> str:
        lines = ["Tracked entities:"]
        
        # Dynamic (tracked)
        for i, name in enumerate(output['dynamic_names']):
            boxes = output['dynamic_boxes'][0, i]  # (T, 4)
            conf = output['dynamic_confidence'][0, i].mean().item()
            
            if conf > 0.3:
                # Show first, middle, last frame boxes
                b0 = boxes[0]
                bm = boxes[self.num_frames // 2]
                bl = boxes[-1]
                
                lines.append(
                    f"  <obj>{name}</obj>:"
                    f"\n    frame 0: <box>[{b0[0]:.2f},{b0[1]:.2f},{b0[2]:.2f},{b0[3]:.2f}]</box>"
                    f"\n    frame {self.num_frames//2}: <box>[{bm[0]:.2f},{bm[1]:.2f},{bm[2]:.2f},{bm[3]:.2f}]</box>"
                    f"\n    frame {self.num_frames-1}: <box>[{bl[0]:.2f},{bl[1]:.2f},{bl[2]:.2f},{bl[3]:.2f}]</box>"
                )
        
        lines.append("\nLocations:")
        
        # Static
        for i, name in enumerate(output['static_names']):
            box = output['static_boxes'][0, i]
            conf = output['static_confidence'][0, i].item()
            
            if conf > 0.3:
                lines.append(
                    f"  <loc>{name}</loc>: "
                    f"<box>[{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}]</box>"
                )
        
        return "\n".join(lines)
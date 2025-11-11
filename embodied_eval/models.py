# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example model wrappers for evaluation."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig

from base import BaseModel


class MockModel(BaseModel):
    """Mock model for testing evaluation framework."""
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize mock model."""
        super().__init__(cfg)
        self.response_mode = self.config.get('response_mode', 'echo')
        self.fixed_response = self.config.get('fixed_response', 'This is a mock answer!')
    
    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """Generate a mock response."""
        # Handle chat format
        if isinstance(prompt, dict):
            prompt_text = prompt.get('content', str(prompt))
        else:
            prompt_text = prompt
        
        if self.response_mode == 'echo':
            # Echo the prompt
            return f"Echo: {prompt_text[:100]}"
        elif self.response_mode == 'fixed':
            return self.fixed_response
        elif self.response_mode == 'random':
            responses = [
                "Yes", "No", "Maybe", 
                "I don't know", "That's interesting",
                "Let me think about that"
            ]
            return random.choice(responses)
        else:
            return "This is a mock answer!"



class Qwen3VLModel(BaseModel):    
    """
    Wrapper for Qwen-3-VL models.
    
    Supports Qwen-3-VL style chat completion format.
    """

    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize Qwen-3-VL model wrapper."""
        super().__init__(cfg)
        
        self.use_chat_format = True  # Qwen-3-VL uses chat format
        
        if not self.model_path:
            raise ValueError("model_path must be specified in config")

    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen-3-VL model.
        
        Args:
            prompt: Text prompt or chat format dict
            images: Optional list of images
            videos: Optional list of videos
            audios: Optional list of audios
            **kwargs: Additional generation arguments
        """
        pass

    
    


class LlamaFactoryModel(BaseModel):
    """
    Wrapper for LlamaFactory models.
    
    This can be extended to support actual model inference using
    the LlamaFactory inference API.
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize LlamaFactory model wrapper."""
        super().__init__(cfg)
        
        # Model-specific settings
        self.use_smart_resize = self.config.get('use_smart_resize', True)
        self.min_pixels = self.config.get('min_pixels', 256)
        self.max_pixels = self.config.get('max_pixels', 1280)
        
        if not self.model_path:
            raise ValueError("model_path must be specified in config")
        
        # TODO: Initialize actual model here
        # from llamafactory.chat import ChatModel
        # self.model = ChatModel(args=...)
    
    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using LlamaFactory model.
        
        Args:
            prompt: Text prompt or chat format dict
            images: Optional list of images
            videos: Optional list of videos
            audios: Optional list of audios
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text response
        """
        # TODO: Implement actual model inference
        # Merge config generation_kwargs with passed kwargs
        gen_kwargs = {**self.generation_kwargs, **kwargs}
        
        # For now, return mock response
        raise NotImplementedError(
            "LlamaFactory model integration not yet implemented. "
            "Use MockModel for testing."
        )
    
    def process_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        target_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """
        Process bounding box for Qwen2.5-VL style models.
        
        Qwen2.5-VL uses smart_resize, so we need to account for that.
        """
        if self.use_smart_resize and format == "qwen_normalized":
            # Apply smart_resize logic
            import math
            
            factor = 28
            min_pixels = self.min_pixels * self.min_pixels
            max_pixels = self.max_pixels * self.max_pixels
            
            width, height = original_size
            h_bar = round(height / factor) * factor
            w_bar = round(width / factor) * factor
            
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = math.floor(height / beta / factor) * factor
                w_bar = math.floor(width / beta / factor) * factor
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = math.ceil(height * beta / factor) * factor
                w_bar = math.ceil(width * beta / factor) * factor
            
            # Normalize bbox to resized dimensions
            scale_x = w_bar / width
            scale_y = h_bar / height
            
            return [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ]
        
        return super().process_bbox(bbox, original_size, target_size, format)


class OpenAIModel(BaseModel):
    """
    Wrapper for OpenAI API models (GPT-4, GPT-4o, etc.).
    
    Supports OpenAI-style chat completion format.
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize OpenAI model wrapper."""
        super().__init__(cfg)
        
        self.api_key = self.config.get('api_key', None)
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        self.use_chat_format = True  # OpenAI always uses chat format
        
        if not self.api_key:
            raise ValueError(
                "api_key must be specified in config or set via OPENAI_API_KEY environment variable"
            )
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError(
                "OpenAI Python package not installed. Install with: pip install openai"
            )
    
    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Text prompt or chat format dict
            images: Optional list of images (for vision models)
            videos: Not supported by OpenAI API
            audios: Not supported by OpenAI API
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text response
        """
        # Merge config generation_kwargs with passed kwargs
        gen_kwargs = {**self.generation_kwargs, **kwargs}
        
        # Build messages
        if isinstance(prompt, dict):
            messages = [prompt]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Add images if provided (for vision models)
        if images:
            # Convert numpy arrays to base64
            import base64
            from io import BytesIO
            from PIL import Image
            
            content = []
            content.append({"type": "text", "text": messages[0]["content"]})
            
            for img in images:
                # Convert numpy array to PIL Image
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img.astype('uint8'))
                else:
                    pil_img = img
                
                # Convert to base64
                buffered = BytesIO()
                pil_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}"
                    }
                })
            
            messages[0]["content"] = content
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_path,  # e.g., "gpt-4o"
            messages=messages,
            **gen_kwargs
        )
        
        return response.choices[0].message.content

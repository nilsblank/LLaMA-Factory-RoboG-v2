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

from base import BaseModel, Sample


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
    
    def prepare_vllm_inputs_from_chat(
        self,
        chat_messages: List[Dict[str, str]],
        sample: Sample
    ) -> Dict[str, Any]:
        """
        Prepare mock vLLM inputs for testing.
        
        Returns a simple tokenized version without actual processing.
        """
        # Simple mock tokenization: just use character codes
        text = " ".join([msg.get("content", "") for msg in chat_messages])
        mock_token_ids = [ord(c) % 1000 for c in text[:100]]  # Mock tokens
        
        # Mock multimodal data
        multi_modal_data = None
        mm_data_dict = {}
        
        if sample.images is not None and len(sample.images) > 0:
            mm_data_dict["image"] = sample.images
        
        if sample.videos is not None and len(sample.videos) > 0:
            mm_data_dict["video"] = sample.videos
        
        if mm_data_dict:
            multi_modal_data = mm_data_dict
        
        return {
            "prompt_token_ids": mock_token_ids,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": None
        }



class Qwen3VLModel(BaseModel):    
    """
    Wrapper for Qwen-3-VL models.
    
    Supports Qwen-3-VL style chat completion format with vLLM inference.
    """

    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize Qwen-3-VL model wrapper."""
        super().__init__(cfg)
        
        self.use_chat_format = True  # Qwen-3-VL uses chat format
        
        if not self.model_path:
            raise ValueError("model_path must be specified in config")
        
        # Initialize processor for tokenization and vision processing
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except ImportError:
            raise ImportError(
                "transformers package not installed. Install with: pip install transformers"
            )
    
    def prepare_inputs_for_vllm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare inputs for vLLM inference.
        
        Args:
            messages: Chat messages in ShareGPT format with content and optional images/videos
            
        Returns:
            Dictionary with:
                - prompt: Tokenized text prompt
                - multi_modal_data: Dict with image/video tensors
                - mm_processor_kwargs: Additional processor kwargs for videos
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen_vl_utils not installed. Install with: pip install qwen-vl-utils"
            )
        
        # Apply chat template to get text prompt
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision inputs (images and videos)
        # qwen_vl_utils 0.0.14+ required
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        # Prepare multimodal data dictionary
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    
    def process_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process prompt and media into Qwen chat message format.
        
        Args:
            prompt: Either a text string or list of chat messages
            images: Optional list of image arrays
            videos: Optional list of video arrays
            
        Returns:
            List of chat messages in Qwen format
        """
        # If prompt is already in message format, use it directly
        if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            return prompt
        
        # Otherwise, build message format from components
        content = []
        
        # Add images
        if images is not None:
            for img in images:
                # Convert numpy array to PIL Image if needed
                if isinstance(img, np.ndarray):
                    from PIL import Image
                    pil_img = Image.fromarray(img.astype('uint8'))
                else:
                    pil_img = img
                
                content.append({
                    "type": "image",
                    "image": pil_img
                })
        
        # Add videos
        if videos is not None:
            for video in videos:
                content.append({
                    "type": "video",
                    "video": video  # Can be path or array
                })
        
        # Add text prompt
        if isinstance(prompt, str):
            content.append({"type": "text", "text": prompt})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages


    def generate(
        self,
        prompt: Union[str, Dict[str, str], List[Dict[str, Any]]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen-3-VL model with vLLM.
        
        Args:
            prompt: Text prompt, chat format dict, or list of messages
            images: Optional list of images
            videos: Optional list of videos
            audios: Optional list of audios (not supported)
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text response
        """
        # Process prompt into message format
        messages = self.process_prompt(prompt, images, videos)
        
        # Prepare inputs for vLLM
        vllm_inputs = self.prepare_inputs_for_vllm(messages)
        
        # Merge config generation_kwargs with passed kwargs
        gen_kwargs = {**self.generation_kwargs, **kwargs}
        
        # Return the prepared inputs for use with vLLM
        # The actual vLLM inference would be handled by the caller
        # For now, we return the inputs structure
        return vllm_inputs
    
    def prepare_vllm_inputs_from_chat(
        self, 
        chat_messages: List[Dict[str, str]], 
        sample: 'Sample'
    ) -> Dict[str, Any]:
        """
        Prepare vLLM inputs from chat messages and sample data.
        
        This method integrates with the benchmark's generate_prompt output
        and prepares inputs in the format expected by vLLM.
        
        Args:
            chat_messages: Chat messages from benchmark.generate_prompt()
            sample: Sample object containing images, videos, metadata
            
        Returns:
            Dictionary with:
                - prompt_token_ids: Tokenized prompt
                - multi_modal_data: Dict with image/video data
                - mm_processor_kwargs: Additional processor kwargs
        """
        # Convert sample images/videos to the format expected by process_vision_info
        # We need to build messages with proper content structure
        messages_with_media = []
        
        for msg in chat_messages:
            content = []
            
            # Parse the text to find <image> tags and build content list
            text = msg.get("content", "")
            
            # Check if text contains <image> tags for interleaving
            image_tag = "<image>"
            
            if image_tag in text and sample.images is not None and len(sample.images) > 0:
                # Interleaved mode: insert images at <image> tag locations
                from PIL import Image
                
                parts = text.split(image_tag)
                image_idx = 0
                
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        content.append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image after text part (except for the last part)
                    if i < len(parts) - 1 and image_idx < len(sample.images):
                        img = sample.images[image_idx]
                        # Convert numpy array to PIL if needed
                        if isinstance(img, np.ndarray):
                            pil_img = Image.fromarray(img.astype('uint8'))
                        else:
                            pil_img = img
                        
                        content.append({
                            "type": "image",
                            "image": pil_img
                        })
                        image_idx += 1
            else:
                # Legacy mode: prepend all images at the start (original behavior)
                # Add images if they exist in the sample
                if sample.images is not None and len(sample.images) > 0 and image_tag in text:
                    from PIL import Image
                    for img in sample.images:
                        # Convert numpy array to PIL if needed
                        if isinstance(img, np.ndarray):
                            pil_img = Image.fromarray(img.astype('uint8'))
                        else:
                            pil_img = img
                        
                        content.append({
                            "type": "image",
                            "image": pil_img
                        })
                
                # Remove <image> tags from text and add text content
                text_cleaned = text.replace(image_tag, "").strip()
                if text_cleaned:
                    content.append({
                        "type": "text",
                        "text": text_cleaned
                    })
            
            # Add videos if they exist in the sample  
            if sample.videos is not None and len(sample.videos) > 0:
                for video in sample.videos:
                    content.append({
                        "type": "video", 
                        "video": video
                    })
            
            #only append user or system messages
            if msg["role"] == "user" or msg["role"] == "system":
                messages_with_media.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })

        # Use prepare_inputs_for_vllm to process
        vllm_inputs = self.prepare_inputs_for_vllm(messages_with_media)
        
        # Add metadata-based processor kwargs
        mm_processor_kwargs = vllm_inputs.get('mm_processor_kwargs', {})
        if sample.metadata:
            if "fps" in sample.metadata:
                mm_processor_kwargs["fps"] = sample.metadata["fps"]
            if "num_frames" in sample.metadata:
                mm_processor_kwargs["num_frames"] = sample.metadata["num_frames"]
        

        return {
            "prompt": vllm_inputs.get('prompt'),
            "multi_modal_data": vllm_inputs.get('multi_modal_data'),
            "mm_processor_kwargs": mm_processor_kwargs if mm_processor_kwargs else None
        }


    
    


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
    
    def prepare_vllm_inputs_from_chat(
        self,
        chat_messages: List[Dict[str, str]],
        sample: Sample
    ) -> Dict[str, Any]:
        """
        Prepare vLLM inputs from chat messages for LlamaFactory models.
        
        This is a generic implementation that should work for most HuggingFace
        models. Override in subclasses for model-specific processing.
        """
        # Try to use transformers AutoTokenizer
        try:
            from transformers import AutoTokenizer
            
            if not hasattr(self, 'tokenizer'):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Apply chat template
            prompt_ids = self.tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Ensure it's a 1D list
            if not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids.tolist()
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            
        except Exception as e:
            # Fallback to simple tokenization
            text = " ".join([msg.get("content", "") for msg in chat_messages])
            prompt_ids = list(range(len(text)))  # Dummy tokens
        
        # Prepare multimodal data
        multi_modal_data = None
        mm_data_dict = {}
        
        if sample.images is not None and len(sample.images) > 0:
            mm_data_dict["image"] = sample.images
        
        if sample.videos is not None and len(sample.videos) > 0:
            mm_data_dict["video"] = sample.videos
        
        if sample.audios is not None and len(sample.audios) > 0:
            mm_data_dict["audio"] = sample.audios
        
        if mm_data_dict:
            multi_modal_data = mm_data_dict
        
        # Prepare processor kwargs from metadata
        mm_processor_kwargs = {}
        if sample.metadata:
            if "fps" in sample.metadata:
                mm_processor_kwargs["fps"] = sample.metadata["fps"]
            if "num_frames" in sample.metadata:
                mm_processor_kwargs["num_frames"] = sample.metadata["num_frames"]
        
        return {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": mm_processor_kwargs if mm_processor_kwargs else None
        }


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
    
    def prepare_vllm_inputs_from_chat(
        self,
        chat_messages: List[Dict[str, str]],
        sample: Sample
    ) -> Dict[str, Any]:
        """
        Prepare vLLM inputs for OpenAI models.
        
        Note: OpenAI models don't use vLLM, so this returns a placeholder.
        This method exists to satisfy the abstract base class requirement.
        """
        # OpenAI doesn't use vLLM, so return a simple structure
        # that indicates this model doesn't support vLLM inference
        text = " ".join([msg.get("content", "") for msg in chat_messages])
        
        return {
            "prompt_token_ids": [],  # OpenAI handles tokenization internally
            "multi_modal_data": None,
            "mm_processor_kwargs": None,
            "_note": "OpenAI models use API-based inference, not vLLM"
        }

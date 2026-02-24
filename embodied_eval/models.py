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

import io
import random
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
import torch

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
        # Determine which tags to look for
        image_tag = "<image>"
        video_tag = "<video>"
        tags_to_find = []
        if sample.images is not None and len(sample.images) > 0:
            tags_to_find.append(image_tag)
        if sample.videos is not None and len(sample.videos) > 0:
            tags_to_find.append(video_tag)

        # Convert sample images/videos to the format expected by process_vision_info
        # We need to build messages with proper content structure
        messages_with_media = []
        image_idx = 0
        video_idx = 0
        
        for msg in chat_messages:
            if msg["role"] != "user" and msg["role"] != "system":
                continue  # Skip other roles

            content = []
            
            # Parse the text to find tags and build content list
            text = msg.get("content", "")
            
            # Split text by tags for interleaving
            if tags_to_find:
                # Create pattern: (<image>|<video>)
                pattern = f"({'|'.join(tags_to_find)})"
                parts = re.split(pattern, text)
            else:
                # No tags to process, treat the whole thing as a single part
                parts = [text]

            for part in parts:
                if part == image_tag:
                    from PIL import Image

                    # Add image if available
                    if image_idx < len(sample.images):
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
                elif part == video_tag:
                    # Add video if available
                    if video_idx < len(sample.videos):
                        video = sample.videos[video_idx]
                        
                        content.append({
                            "type": "video",
                            "video": video
                        })
                        video_idx += 1
                else:
                    # Add text part if not empty
                    if part.strip():
                        content.append({
                            "type": "text",
                            "text": part
                        })

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

    def denormalize_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """Qwen 3 VL uses coordinates in range 0-1000 for bounding boxes by default."""
    
        if format == "xyxy":
            width, height = original_size
            x1 = bbox[0] / 1000 * width
            y1 = bbox[1] / 1000 * height
            x2 = bbox[2] / 1000 * width
            y2 = bbox[3] / 1000 * height
            return [x1, y1, x2, y2]
        else:
            raise NotImplementedError()


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
        self.system_prompt = "When returning bounding boxes, use normalized image coordinates in the range 0.0 to 1.0, formatted as (x_min, y_min, x_max, y_max), where (0,0) is the top-left and (1,1) is the bottom-right of the image."
        self.use_chat_format = True  # OpenAI always uses chat format
        self.max_attempts = self.config.get('max_attempts', 3)
        self.number_video_frames = self.config.get('number_video_frames', 16)
        
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
    
    def _encode_image(self, image_input):
        """
        Encodes an image from a file path, PIL Image, or Numpy array to a Base64 string.
        
        Args:
            image_input: Union[str, PIL.Image.Image, np.ndarray]
        """
        import base64
        from PIL import Image

        # 1. Handle String (File Path)
        if isinstance(image_input, str):
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # 2. Handle Numpy Array
        if isinstance(image_input, np.ndarray):
            image_input = Image.fromarray(image_input.astype('uint8'))

        # 3. Handle PIL Image
        if isinstance(image_input, Image.Image):
            buffered = io.BytesIO()            
            image_input.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    def _encode_video(self, video_input):
        """
        Encodes a video from a file path to a list of Base64-encoded frames,
        downsampling to self.number_video_frames if needed.

        Args:
            video_input: str, path to video file

        Returns:
            List[str]: Base64-encoded frames
        """
        import base64
        import cv2

        if not isinstance(video_input, str):
            raise ValueError(f"Expected video file path as string, got {type(video_input)}")

        video = cv2.VideoCapture(video_input)
        if not video.isOpened():
            raise ValueError(f"Failed to open video file: {video_input}")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            video.release()
            return []

        # Frames to keep
        frame_indices = set(np.linspace(0, total_frames - 1, min(self.number_video_frames, total_frames), dtype=int))

        frames = []
        for idx in range(total_frames):
            success, frame = video.read()
            if not success:
                break
            if idx in frame_indices:
                ret, buffer = cv2.imencode(".png", frame)
                if ret:
                    frames.append(base64.b64encode(buffer).decode("utf-8"))
            if len(frames) == len(frame_indices):
                break

        video.release()
        return frames

    def denormalize_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """
        By default the system prompt tells the models to use normalized coordinates from 0.0 to 1.0 for bounding boxes.
        """
        if format == "xyxy":
            width, height = original_size
            x1 = bbox[0] * width
            y1 = bbox[1] * height
            x2 = bbox[2] * width
            y2 = bbox[3] * height
            return [x1, y1, x2, y2]
        else:
            raise NotImplementedError()

    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[str]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Text prompt or chat format dict
            images: Optional list of images
            videos: Optional list of videos as file path
            audios: Not supported by OpenAI API
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text response
        """
        # Merge config generation_kwargs with passed kwargs
        gen_kwargs = {**self.generation_kwargs, **kwargs}
        
        # Build messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise NotImplementedError("Unsupported prompt format")
        
        # OpenAI API only supports images right now
        if audios is not None and len(audios) > 0:
            raise NotImplementedError("OpenAI API does not support audio inputs")

        # Determine which tags to look for
        image_tag = "<image>"
        video_tag = "<video>"
        tags_to_find = []
        if images is not None and len(images) > 0:
            tags_to_find.append(image_tag)
        if videos is not None and len(videos) > 0:
            tags_to_find.append(video_tag)

        # Add images and videos
        messages_with_media = []
        image_idx = 0
        video_idx = 0
        
        for msg in messages:
            if msg["role"] != "user" and msg["role"] != "system":
                continue  # Skip other roles

            content = []
            
            # Parse the text to find tags and build content list
            text = msg.get("content", "")
            
            # Split text by tags for interleaving
            if tags_to_find:
                # Create pattern: (<image>|<video>)
                pattern = f"({'|'.join(tags_to_find)})"
                parts = re.split(pattern, text)
            else:
                # No tags to process, treat the whole thing as a single part
                parts = [text]

            for part in parts:
                if part == image_tag:
                    # Add image if available
                    if image_idx < len(images):
                        img = images[image_idx]
                        # Convert to base64
                        img_str = self._encode_image(img)
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        })
                        image_idx += 1
                elif part == video_tag:
                    # Add video if available
                    if video_idx < len(videos):
                        video = videos[video_idx]
                        # Convert to list of base64
                        video_str = self._encode_video(video)
                        # Sample fix amount of images
                        if len(video_str) > self.number_video_frames:
                            video_str = [video_str[i] for i in np.linspace(0, len(video_str)-1, self.number_video_frames, dtype=int)]

                        content.extend([{
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        } for img_str in video_str])
                        video_idx += 1
                else:
                    # Add text part if not empty
                    if part.strip():
                        content.append({
                            "type": "text",
                            "text": part
                        })

            # Append the model's own system prompt to clarify bbox coordinates
            if msg["role"] == "system":
                content.append({
                    "type": "text",
                    "text": self.system_prompt
                })

            messages_with_media.append({
                "role": msg.get("role", "user"),
                "content": content
            })
        
        # Call OpenAI API
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages_with_media,
                    **gen_kwargs
                )
                break  # If successful, break the loop
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    raise e # Re-raise error on the last attempt
                time.sleep(2 ** attempt)
        
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


class GoogleModel(BaseModel):
    """
    Wrapper for Google's API models (Gemini 3 Pro, etc.).
    
    Uses Google's genai interface.
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize Google model wrapper."""
        super().__init__(cfg)
        
        self.api_key = self.config.get('api_key', None)
        self.system_prompt = "When returning bounding boxes, use normalized image coordinates in the range 0 to 1000, formatted as (x_min, y_min, x_max, y_max), where (0,0) is the top-left and (1000,1000) is the bottom-right of the image."
        self.use_chat_format = True
        self.thinking_level = self.config.get('thinking_level', "low")
        self.max_attempts = self.config.get('max_attempts', 3)
        
        if not self.api_key:
            raise ValueError(
                "api_key must be specified in config or set via OPENAI_API_KEY environment variable"
            )
        
        # Initialize genai client
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Google Python package not installed. Install with: pip install google-genai"
            )
    
    def _encode_image(self, image_input):
        """
        Encodes an image from a file path, PIL Image, or Numpy array to bytes.
        
        Args:
            image_input: Union[str, PIL.Image.Image, np.ndarray]
        """
        from PIL import Image

        # 1. Handle String (File Path)
        if isinstance(image_input, str):
            with open(image_input, 'rb') as f:
                return f.read()

        # 2. Handle Numpy Array
        if isinstance(image_input, np.ndarray):
            image_input = Image.fromarray(image_input.astype('uint8'))

        # 3. Handle PIL Image
        if isinstance(image_input, Image.Image):
            buffered = io.BytesIO()            
            image_input.save(buffered, format="JPEG")
            return buffered.getvalue()

        raise ValueError(f"Unsupported image type: {type(image_input)}")

    def denormalize_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """
        By default the system prompt tells the models to use normalized coordinates from 0 to 1000 for bounding boxes.
        This should match the training setup of the Gemini 3 models.
        """
        if format == "xyxy":
            width, height = original_size
            x1 = bbox[0] / 1000 * width
            y1 = bbox[1] / 1000 * height
            x2 = bbox[2] / 1000 * width
            y2 = bbox[3] / 1000 * height
            return [x1, y1, x2, y2]
        else:
            raise NotImplementedError()

    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[str]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using genai API.
        
        Args:
            prompt: Text prompt or chat format dict
            images: Optional list of images as np.ndarray PIL.Image or file path
            videos: Optional list of videos as file path
            audios: Not implemented
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text response
        """
        from google.genai import types

        # Merge config generation_kwargs with passed kwargs
        gen_kwargs = {**self.generation_kwargs, **kwargs}
        
        # Build messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise NotImplementedError("Unsupported prompt format")
        
        # Audio is not implemented yet
        if audios is not None and len(audios) > 0:
            raise NotImplementedError("Supported but not implemented yet")

        # Determine which tags to look for
        image_tag = "<image>"
        video_tag = "<video>"
        tags_to_find = []
        if images is not None and len(images) > 0:
            tags_to_find.append(image_tag)
        if videos is not None and len(videos) > 0:
            tags_to_find.append(video_tag)

        # Add images and videos
        contents = []
        system_instruction = ""
        image_idx = 0
        video_idx = 0
        
        for msg in messages:
            if msg["role"] == "system":
                text = msg.get("content", "")
                system_instruction += text
            elif msg["role"] == "user":
                # Parse the text to find tags and build content list
                text = msg.get("content", "")
                
                # Split text by tags for interleaving
                if tags_to_find:
                    # Create pattern: (<image>|<video>)
                    pattern = f"({'|'.join(tags_to_find)})"
                    parts = re.split(pattern, text)
                else:
                    # No tags to process, treat the whole thing as a single part
                    parts = [text]

                for part in parts:
                    if part == image_tag:
                        # Add image if available
                        if image_idx < len(images):
                            img = images[image_idx]
                            # Convert to bytes
                            img_bytes = self._encode_image(img)
                            
                            contents.append(types.Part.from_bytes(
                                data=img_bytes,
                                mime_type='image/jpeg',
                            ),)
                            image_idx += 1
                    elif part == video_tag:
                        # Add video if available
                        if video_idx < len(videos):
                            # Upload video to Google cloud
                            video = videos[video_idx]
                            if not isinstance(video, str):
                                raise ValueError(
                                    f"Unsupported video type: {type(video)}"
                                )
                            video_file = self.client.files.upload(file=video)
                            # Poll until upload is complete
                            seconds_waited = 0
                            while video_file.state.name == "PROCESSING":
                                if seconds_waited > 30:
                                    raise TimeoutError(
                                        "Video upload timed out after 30 seconds"
                                    )
                                time.sleep(1)  # Wait 1 seconds before checking again
                                seconds_waited += 1
                                video_file = self.client.files.get(name=video_file.name)

                            contents.append(video_file)
                            video_idx += 1
                    else:
                        # Add text part if not empty
                        if part.strip():
                            contents.append(part)

            else:  # Only collect user or system messages
                continue

        # Call Google API
        for attempt in range(self.max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_path,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction + self.system_prompt,
                        thinkingConfig=types.ThinkingConfig(thinking_level=self.thinking_level),
                        **gen_kwargs
                    )
                )
                break  # If successful, break the loop
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    raise e # Re-raise error on the last attempt
                time.sleep(2 ** attempt)
        
        return response.text
    
    def prepare_vllm_inputs_from_chat(
        self,
        chat_messages: List[Dict[str, str]],
        sample: Sample
    ) -> Dict[str, Any]:
        """
        Prepare vLLM inputs for Google models.
        
        Note: Google models don't use vLLM, so this returns a placeholder.
        This method exists to satisfy the abstract base class requirement.
        """
        # Google doesn't use vLLM, so return a simple structure
        # that indicates this model doesn't support vLLM inference
        text = " ".join([msg.get("content", "") for msg in chat_messages])
        
        return {
            "prompt_token_ids": [],  # Google handles tokenization internally
            "multi_modal_data": None,
            "mm_processor_kwargs": None,
            "_note": "Google models use API-based inference, not vLLM"
        }

class RoboAnnotatorX(BaseModel):
    """
    Wrapper for the RoboAnnotatorX model to load the components
    and run inference.
    """

    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """Initialize RoboAnnotatorX wrapper."""
        super().__init__(cfg)

        # Optional base model name used by the example loader
        self.model_base = self.config.get("model_base", None)
        # Device to run inference on
        self.device = self.config.get("device", "cuda")
        # System Prompt
        self.system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        # Components
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self._load_model()

    def _load_model(self):
        """Load tokenizer, model and image processor."""
        try:
            from roboannotatorx.model.builder import load_roboannotator
        except Exception as e:
            raise ImportError(
                "roboannotatorx package not installed or failed to import. "
                "Install and ensure it's importable. Original error: {}".format(e)
            )

        # The example returns tokenizer, model, image_processor, _
        tokenizer, model, image_processor, _ = load_roboannotator(
            model_path=self.model_path,
            model_base=self.model_base,
        )

        # Move model to device if possible
        try:
            model = model.to(self.device)
        except Exception:
            pass

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    def _load_image(self, image_input, num_frames=8):
        from PIL import Image

        # Handle file path
        if isinstance(image_input, str):
            image_input = Image.open(image_input).convert("RGB")

        # Handle PIL Image
        if isinstance(image_input, Image.Image):
            frame = np.array(image_input)   # (H, W, C)
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        
        # TODO Try to match this:
        # if type(images) is list or images.ndim == 5: # video
        #     images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
        #     image_counts = [image.shape[0] for image in images]
        #     concat_images = torch.cat(images, dim=0)
        #     image_features = self.encode_images(concat_images, image_counts)
        # else: # single-image / multi-image
        #     image_features = self.encode_images(images)
        
        # Fix by repeating image
        frames = np.repeat(frame[None, ...], num_frames, axis=0)  # (T, H, W, C)

        # Prepare images
        tensor = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]  # (1, C, H, W)

        return tensor.half().to(self.device)

    def _load_video(self, video_input: str):
        from decord import VideoReader, cpu

        if not isinstance(video_input, str):
            raise ValueError(f"Expected video file path as string, got {type(video_input)}")

        vr = VideoReader(video_input, ctx=cpu(0))
        frame_idx = list(range(len(vr)))

        # Downsampling
        if len(frame_idx) > 2000:
            frame_idx = frame_idx[::4]
        elif len(frame_idx) > 500:
            frame_idx = frame_idx[::2]

        frames = vr.get_batch(frame_idx).asnumpy()
        tensor = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]
        return tensor.half().to(self.device)

    def _build_prompt(self, messages, images, videos):
        from roboannotatorx.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN
        )

        # Determine which tags to look for
        image_tag = "<image>"
        video_tag = "<video>"
        tags_to_find = []
        if images is not None and len(images) > 0:
            tags_to_find.append(image_tag)
        if videos is not None and len(videos) > 0:
            tags_to_find.append(video_tag)

        # Process messages
        processed_prompt = ""
        combined_media = []
        image_idx = 0
        video_idx = 0
        media_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if self.model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        seps = [" ", "</s>"]
        sep_idx = 0

        for msg in messages:
            role = msg["role"]
            if role != "user" and role != "system":
                continue  # Skip other roles
            
            # Parse the text to find tags and build content
            text = msg.get("content", "")
            content = ""
            
            # Split text by tags for interleaving
            if tags_to_find:
                # Create pattern: (<image>|<video>)
                pattern = f"({'|'.join(tags_to_find)})"
                parts = re.split(pattern, text)
            else:
                # No tags to process, treat the whole thing as a single part
                parts = [text]

            for part in parts:
                if part == image_tag:
                    # Add image if available
                    if image_idx < len(images):
                        combined_media.append(self._load_image(images[image_idx]))
                        content += f"{media_token}\n"
                        image_idx += 1
                elif part == video_tag:
                    # Add video if available
                    if video_idx < len(videos):
                        combined_media.append(self._load_video(videos[video_idx]))
                        content += f"{media_token}\n"
                        video_idx += 1
                else:
                    # Add text part if not empty
                    if part.strip():
                        content += f"{part}\n"

            if role == "system":
                # Append the model's own system prompt
                content += self.system_prompt
                # Add message to processed prompt
                processed_prompt += content.strip() + seps[0]
            else: 
                # Add message to processed prompt
                sep = seps[sep_idx % 2]
                processed_prompt += role.upper() + ": " + content.strip() + sep
                sep_idx += 1

        processed_prompt += "ASSISTANT:"
        return processed_prompt, combined_media

    def generate(
        self,
        prompt: Union[str, Dict[str, str], List[Dict[str, str]]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[str]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """Run RoboAnnotatorX inference and return decoded text.

        The implementation follows the example usage at the bottom of this file.
        """
        from roboannotatorx.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from roboannotatorx.constants import IMAGE_TOKEN_INDEX
        
        # Merge generation kwargs
        gen_kwargs = {**self.generation_kwargs, **kwargs}

        # Build messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise NotImplementedError("Unsupported prompt format")
        
        # Audio is not supported
        if audios is not None and len(audios) > 0:
            raise NotImplementedError("RoboAnnotatorX does not support audio inputs")

        # Build prompt
        processed_prompt, combined_media = self._build_prompt(messages, images, videos)

        # Tokenize and insert image token using helper if available
        input_ids = tokenizer_image_token(
            processed_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        # Stopping criteria
        keywords = ["</s>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Run inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=combined_media,
                do_sample=True,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **gen_kwargs,
            )

        # Decode output
        output = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        return output

    def denormalize_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """Assume RoboAnnotatorX returns normalized 0.0-1.0 coords by default.

        Convert to pixel coordinates in the same style as OpenAIModel.
        """
        # TODO verify this matches training setup, unclear if they actually trained bbox prediction
        if format == "xyxy":
            width, height = original_size
            x1 = bbox[0] * width
            y1 = bbox[1] * height
            x2 = bbox[2] * width
            y2 = bbox[3] * height
            return [x1, y1, x2, y2]
        else:
            raise NotImplementedError()

    def prepare_vllm_inputs_from_chat(
        self,
        chat_messages: List[Dict[str, str]],
        sample: Sample
    ) -> Dict[str, Any]:
        """
        Note: RoboAnnotatorX doesn't use vLLM, so this returns a placeholder.
        This method exists to satisfy the abstract base class requirement.
        """
        # RoboAnnotatorX doesn't use vLLM, so return a simple structure
        # that indicates this model doesn't support vLLM inference
        text = " ".join([msg.get("content", "") for msg in chat_messages])
        
        return {
            "prompt_token_ids": [],
            "multi_modal_data": None,
            "mm_processor_kwargs": None,
            "_note": "RoboAnnotatorX does not use vLLM"
        }

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

"""Utility functions for the evaluation framework."""

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int, int]:
    """
    Qwen2.5-VL smart resize function.
    
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels are within [min_pixels, max_pixels]
    3. Aspect ratio is maintained as closely as possible
    
    Args:
        height: Original height
        width: Original width
        factor: Divisibility factor (default: 28)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        
    Returns:
        Tuple of (new_height, new_width)
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    
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
    
    return h_bar, w_bar


def draw_bbox_on_image(
    image: Union[np.ndarray, Image.Image],
    bbox: List[float],
    label: Optional[str] = None,
    color: str = 'red',
    width: int = 3,
    input_width: Optional[int] = None,
    input_height: Optional[int] = None
) -> Image.Image:
    """
    Draw bounding box on image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        label: Optional label text
        color: Box color
        width: Line width
        input_width: Original input width (for denormalization)
        input_height: Original input height (for denormalization)
        
    Returns:
        PIL Image with drawn bbox
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    else:
        image = image.copy()
    
    img_width, img_height = image.size
    
    # Denormalize if needed
    if input_width is not None and input_height is not None:
        scale_x = img_width / input_width
        scale_y = img_height / input_height
        
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)
    else:
        x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Draw bbox
    draw = ImageDraw.Draw(image)
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)
    
    # Draw label if provided
    if label is not None:
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except:
            font = ImageFont.load_default()
        
        # Draw background for text
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill='white', font=font)
    
    return image


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Determine intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    if union == 0:
        return 0.0
    return intersection / union


def normalize_bbox(
    bbox: List[float],
    width: int,
    height: int,
    format: str = 'xyxy'
) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1].
    
    Args:
        bbox: Bounding box coordinates
        width: Image width
        height: Image height
        format: Bbox format ('xyxy', 'xywh', 'cxcywh')
        
    Returns:
        Normalized bbox
    """
    if format == 'xyxy':
        return [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]
    elif format == 'xywh':
        return [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]
    elif format == 'cxcywh':
        return [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]
    else:
        raise ValueError(f"Unknown format: {format}")


def denormalize_bbox(
    bbox: List[float],
    width: int,
    height: int,
    format: str = 'xyxy'
) -> List[float]:
    """
    Denormalize bounding box coordinates from [0, 1].
    
    Args:
        bbox: Normalized bounding box coordinates
        width: Image width
        height: Image height
        format: Bbox format ('xyxy', 'xywh', 'cxcywh')
        
    Returns:
        Denormalized bbox
    """
    if format == 'xyxy':
        return [
            bbox[0] * width,
            bbox[1] * height,
            bbox[2] * width,
            bbox[3] * height
        ]
    elif format == 'xywh':
        return [
            bbox[0] * width,
            bbox[1] * height,
            bbox[2] * width,
            bbox[3] * height
        ]
    elif format == 'cxcywh':
        return [
            bbox[0] * width,
            bbox[1] * height,
            bbox[2] * width,
            bbox[3] * height
        ]
    else:
        raise ValueError(f"Unknown format: {format}")


def convert_bbox_format(
    bbox: List[float],
    from_format: str,
    to_format: str
) -> List[float]:
    """
    Convert bounding box between different formats.
    
    Args:
        bbox: Input bbox
        from_format: Current format ('xyxy', 'xywh', 'cxcywh')
        to_format: Target format ('xyxy', 'xywh', 'cxcywh')
        
    Returns:
        Converted bbox
    """
    # First convert to xyxy
    if from_format == 'xyxy':
        xyxy = bbox
    elif from_format == 'xywh':
        xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    elif from_format == 'cxcywh':
        cx, cy, w, h = bbox
        xyxy = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    else:
        raise ValueError(f"Unknown from_format: {from_format}")
    
    # Then convert to target format
    if to_format == 'xyxy':
        return xyxy
    elif to_format == 'xywh':
        return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
    elif to_format == 'cxcywh':
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        return [xyxy[0] + w/2, xyxy[1] + h/2, w, h]
    else:
        raise ValueError(f"Unknown to_format: {to_format}")



def share_gpt_to_hf_message(item: Dict[str, Any]) -> str:
    """
    Converts a sharGPT formated message to Hugging Face format.
    """
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": img} for img in images
    ]
    video_pool = [
        {"type": "video", "video": vid} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages   

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating how to use the Qwen3VLModel class with vLLM inference.

This example shows:
1. How to initialize the model
2. How to process prompts and prepare inputs for vLLM
3. How to integrate with the benchmark's generate_prompt workflow
"""

import sys
from pathlib import Path

# Add embodied_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Qwen3VLModel
from base import Sample
import numpy as np
from PIL import Image


def example_1_basic_usage():
    """Example 1: Basic usage with text and images."""
    print("=" * 60)
    print("Example 1: Basic usage with text and images")
    print("=" * 60)
    
    # Initialize model
    config = {
        "name": "qwen3-vl-8b",
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "generation_kwargs": {
            "temperature": 0,
            "max_tokens": 1024,
            "top_k": -1
        }
    }
    
    model = Qwen3VLModel(cfg=config)
    
    # Create a sample image (dummy for demo)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Option A: Use process_prompt to build messages
    messages = model.process_prompt(
        prompt="What is in this image?",
        images=[dummy_image]
    )
    
    print(f"\nProcessed messages structure:")
    print(f"  Number of messages: {len(messages)}")
    print(f"  First message role: {messages[0]['role']}")
    print(f"  Content items: {len(messages[0]['content'])}")
    
    # Option B: Prepare inputs for vLLM
    vllm_inputs = model.prepare_inputs_for_vllm(messages)
    
    print(f"\nvLLM inputs:")
    print(f"  Prompt length: {len(vllm_inputs['prompt'])}")
    print(f"  Multi-modal data keys: {list(vllm_inputs['multi_modal_data'].keys())}")
    print(f"  MM processor kwargs: {vllm_inputs['mm_processor_kwargs']}")


def example_2_benchmark_integration():
    """Example 2: Integration with benchmark workflow."""
    print("\n" + "=" * 60)
    print("Example 2: Integration with benchmark workflow")
    print("=" * 60)
    
    # Initialize model
    config = {
        "name": "qwen3-vl-8b",
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "generation_kwargs": {
            "temperature": 0,
            "max_tokens": 1024
        }
    }
    
    model = Qwen3VLModel(cfg=config)
    
    # Create a sample (as would come from benchmark)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sample = Sample(
        question="Describe this image in detail.",
        answer="This is a test answer.",
        images=[dummy_image],
        metadata={"image_id": "test_001"}
    )
    
    # Simulate what generate_prompt returns (chat messages with <image> tags)
    chat_messages = [
        {
            "role": "user",
            "content": "<image>\nDescribe this image in detail."
        },
        {
            "role": "assistant", 
            "content": "This is a test answer."
        }
    ]
    
    # Use the new method to prepare vLLM inputs
    vllm_inputs = model.prepare_vllm_inputs_from_chat(chat_messages, sample)
    
    print(f"\nvLLM inputs from chat:")
    print(f"  Prompt token IDs: {len(vllm_inputs['prompt_token_ids'])} tokens")
    print(f"  Multi-modal data: {vllm_inputs['multi_modal_data'] is not None}")
    if vllm_inputs['multi_modal_data']:
        print(f"    - Image data present: {'image' in vllm_inputs['multi_modal_data']}")
        print(f"    - Video data present: {'video' in vllm_inputs['multi_modal_data']}")
    print(f"  MM processor kwargs: {vllm_inputs['mm_processor_kwargs']}")


def example_3_video_with_metadata():
    """Example 3: Video processing with metadata (fps, num_frames)."""
    print("\n" + "=" * 60)
    print("Example 3: Video processing with metadata")
    print("=" * 60)
    
    # Initialize model
    config = {
        "name": "qwen3-vl-8b",
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct"
    }
    
    model = Qwen3VLModel(cfg=config)
    
    # Create a sample with video and metadata
    # (In real usage, video would be a path or video tensor)
    sample = Sample(
        question="What happens in this video?",
        answer="A person is walking.",
        videos=["path/to/video.mp4"],  # Or actual video tensor
        metadata={
            "video_id": "vid_001",
            "fps": 4,
            "num_frames": 16
        }
    )
    
    # Chat messages from generate_prompt
    chat_messages = [
        {
            "role": "user",
            "content": "What happens in this video?"
        }
    ]
    
    # Prepare vLLM inputs
    vllm_inputs = model.prepare_vllm_inputs_from_chat(chat_messages, sample)
    
    print(f"\nvLLM inputs with video:")
    print(f"  Prompt token IDs: {len(vllm_inputs['prompt_token_ids'])} tokens")
    print(f"  Multi-modal data: {vllm_inputs['multi_modal_data'] is not None}")
    if vllm_inputs['multi_modal_data']:
        print(f"    - Video data present: {'video' in vllm_inputs['multi_modal_data']}")
    print(f"  MM processor kwargs: {vllm_inputs['mm_processor_kwargs']}")
    if vllm_inputs['mm_processor_kwargs']:
        print(f"    - FPS: {vllm_inputs['mm_processor_kwargs'].get('fps')}")
        print(f"    - Num frames: {vllm_inputs['mm_processor_kwargs'].get('num_frames')}")


def example_4_vllm_inference_flow():
    """Example 4: Complete vLLM inference flow."""
    print("\n" + "=" * 60)
    print("Example 4: Complete vLLM inference flow (pseudocode)")
    print("=" * 60)
    
    print("""
    # Initialize vLLM and model
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=1,
        seed=0
    )
    
    model = Qwen3VLModel(cfg={
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct"
    })
    
    # Process benchmark samples
    for sample in benchmark.samples:
        # Get chat messages from benchmark
        chat_messages = benchmark.generate_prompt(sample, model)
        
        # Use model to prepare vLLM inputs
        vllm_input = model.prepare_vllm_inputs_from_chat(chat_messages, sample)
        
        # Add to batch
        batch_inputs.append(vllm_input)
    
    # Generate with vLLM
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1
    )
    
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    
    # Process outputs
    for output in outputs:
        generated_text = output.outputs[0].text
        # Apply model-specific parsing if needed
        parsed = model.parse_output(generated_text)
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Qwen3VLModel Usage Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
    except Exception as e:
        print(f"\nExample 1 failed (expected if dependencies not installed): {e}")
    
    try:
        example_2_benchmark_integration()
    except Exception as e:
        print(f"\nExample 2 failed (expected if dependencies not installed): {e}")
    
    try:
        example_3_video_with_metadata()
    except Exception as e:
        print(f"\nExample 3 failed (expected if dependencies not installed): {e}")
    
    example_4_vllm_inference_flow()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

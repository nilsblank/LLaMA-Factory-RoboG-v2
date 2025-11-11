"""
Example: Using vllm_infer.py with benchmark framework.

This shows the STANDARD way to run inference - using the vllm_infer.py script
which is optimized for batch processing.
"""

import subprocess
import sys

def run_vllm_inference():
    """
    Run vLLM inference using the benchmark framework.
    
    The vllm_infer.py script is the standard way to run inference because:
    1. Uses optimized vLLM batch processing
    2. Works seamlessly with benchmark classes
    3. Automatic evaluation
    4. Efficient multi-GPU support
    """
    
    # Example 1: Using benchmark config file
    cmd1 = [
        sys.executable, "-m", "embodied_eval.vllm_infer",
        "--model_name_or_path", "/path/to/qwen2.5-vl-7b-instruct",
        "--template", "qwen2_5_vl",
        "--benchmark_config", "embodied_eval/configs/benchmarks/robovqa.yaml",
        "--batch_size", "1024",
        "--output_dir", "./vllm_results",
    ]
    
    print("Example 1: Using benchmark config file")
    print(" ".join(cmd1))
    print()
    
    # Example 2: Using benchmark name with overrides
    cmd2 = [
        sys.executable, "-m", "embodied_eval.vllm_infer",
        "--model_name_or_path", "/path/to/model",
        "--template", "default",
        "--benchmark_name", "robovqa",
        "--benchmark_data_dir", "/path/to/robovqa/tfrecords",
        "--benchmark_split", "validation",
        "--batch_size", "1024",
        "--max_samples", "100",  # Quick test
    ]
    
    print("Example 2: Using benchmark name with overrides")
    print(" ".join(cmd2))
    print()
    
    # Example 3: Multi-GPU with LoRA
    cmd3 = [
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        sys.executable, "-m", "embodied_eval.vllm_infer",
        "--model_name_or_path", "/path/to/base/model",
        "--adapter_name_or_path", "/path/to/lora/adapter",
        "--template", "qwen2_5_vl",
        "--benchmark_config", "embodied_eval/configs/benchmarks/robovqa.yaml",
        "--batch_size", "2048",
        "--pipeline_parallel_size", "1",
    ]
    
    print("Example 3: Multi-GPU with LoRA")
    print(" ".join(cmd3))
    print()
    
    # Example 4: With custom vLLM config
    cmd4 = [
        sys.executable, "-m", "embodied_eval.vllm_infer",
        "--model_name_or_path", "/path/to/model",
        "--benchmark_name", "robovqa",
        "--vllm_config", "{'gpu_memory_utilization': 0.9, 'quantization': 'awq'}",
        "--batch_size", "1024",
    ]
    
    print("Example 4: With custom vLLM config")
    print(" ".join(cmd4))
    print()
    
    print("="*70)
    print("To actually run, uncomment one of the subprocess.run() calls below")
    print("="*70)
    
    # Uncomment to actually run:
    # subprocess.run(cmd1)
    # subprocess.run(cmd2)
    # subprocess.run(cmd3, shell=True)  # Note: shell=True for env vars
    # subprocess.run(cmd4)


if __name__ == "__main__":
    run_vllm_inference()

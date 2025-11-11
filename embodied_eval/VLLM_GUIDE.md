# vLLM Integration Guide

## Overview

**`vllm_infer.py` is the STANDARD way to run inference** in this framework. It combines:
- ✅ vLLM's efficient batch processing and tensor parallelism
- ✅ Seamless benchmark loading from the evaluation framework
- ✅ Automatic evaluation after inference
- ✅ Simple command-line interface

This script is adapted from LlamaFactory's `scripts/vllm_infer.py` to work with the benchmark framework.

## Quick Start

### Basic Usage
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/your/model \
    --template qwen2_5_vl \
    --benchmark_name robovqa \
    --batch_size 1024
```

### Using a Benchmark Config File
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_config embodied_eval/configs/benchmarks/robovqa.yaml \
    --batch_size 1024
```

## Key Features

### 1. **Benchmark Integration**
The script automatically:
- Loads benchmarks using `BENCHMARK_REGISTRY`
- Uses `benchmark.preprocess()` for sample preprocessing
- Calls `benchmark.generate_prompt()` for prompt generation
- Runs `benchmark.evaluate()` after inference
- Saves predictions and results

### 2. **Efficient Batch Processing**
- Uses vLLM's native batch processing (same as original script)
- Automatically adjusts batch size based on GPU memory
- Supports multi-GPU tensor parallelism

### 3. **Full vLLM Feature Support**
- LoRA adapters
- Multi-modal models (vision, audio)
- Custom sampling parameters
- Quantization (GPTQ, AWQ, etc.)

## Command-Line Arguments

### Required Arguments
```bash
--model_name_or_path PATH    # Path to model or HuggingFace ID
--benchmark_name NAME        # Name of benchmark (e.g., "robovqa")
# OR
--benchmark_config PATH      # Path to benchmark YAML config
```

### Common Options
```bash
--template NAME              # Chat template (qwen2_5_vl, llama3, etc.)
--batch_size N               # Batch size for inference (default: 1024)
--max_samples N              # Limit number of samples (for testing)
--output_dir PATH            # Where to save results
```

### vLLM Options
```bash
--adapter_name_or_path PATH  # LoRA adapter path
--tensor_parallel_size N     # Number of GPUs for tensor parallelism
--gpu_memory_utilization F   # GPU memory usage (0.0-1.0, default: 0.9)
--max_model_len N            # Max sequence length
--quantization_method NAME   # Quantization (gptq, awq, etc.)
```

### Generation Options
```bash
--temperature F              # Sampling temperature (default: 0.0)
--top_p F                    # Nucleus sampling threshold
--top_k N                    # Top-k sampling
--max_tokens N               # Max tokens to generate
```

## Usage Examples

### 1. Quick Test (10 Samples)
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_name robovqa \
    --max_samples 10 \
    --batch_size 10
```

### 2. Multi-GPU Inference
```bash
# Automatically uses tensor parallelism for large models
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/70B/model \
    --benchmark_name robovqa \
    --batch_size 2048
```

### 3. LoRA Adapter
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/base_model \
    --adapter_name_or_path /path/to/lora_adapter \
    --benchmark_name robovqa \
    --batch_size 1024
```

### 4. Custom Benchmark Config
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_config /path/to/my_benchmark.yaml \
    --batch_size 512 \
    --temperature 0.7
```

### 5. Specific GPU
```bash
CUDA_VISIBLE_DEVICES=2 python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_name robovqa \
    --batch_size 1024
```

## How It Works

### 1. Load Benchmark
```python
# Loads from registry using benchmark_name or config file
benchmark_cls = BENCHMARK_REGISTRY.get(benchmark_name)
benchmark = benchmark_cls(config)
```

### 2. Prepare Data
```python
# Uses benchmark's preprocessing and prompt generation
for sample in benchmark:
    processed = benchmark.preprocess(sample)
    messages = benchmark.generate_prompt(processed, model)
```

### 3. Run vLLM Inference
```python
# Efficient batch processing (same as original vllm_infer.py)
outputs = llm.generate(prompts, sampling_params)
```

### 4. Evaluate
```python
# Automatic evaluation using benchmark's metrics
results = benchmark.evaluate(predictions, ground_truths, metadata)
benchmark.print_summary()
```

## Output Files

The script saves:
- `generated_predictions.jsonl`: Model predictions in JSONL format
- `eval_results.json`: Evaluation metrics
- Console output: Summary statistics and sample predictions

## Performance Tips

### Batch Size
- Start with 1024 and adjust based on GPU memory
- Larger batch = faster inference (up to memory limit)
- Monitor GPU memory: `nvidia-smi`

### Multi-GPU
- vLLM automatically uses tensor parallelism
- Set `CUDA_VISIBLE_DEVICES` to control which GPUs to use
- More GPUs = larger models + faster inference

### Memory Optimization
```bash
# Reduce memory usage
--gpu_memory_utilization 0.8 \
--max_model_len 4096
```

## Comparison: vllm_infer.py vs run_eval.py

| Feature | vllm_infer.py | run_eval.py |
|---------|---------------|-------------|
| **Inference** | vLLM (FAST) | BaseModel classes |
| **Use Case** | Single benchmark, vLLM models | Multi-benchmark, multi-model |
| **Speed** | ⚡ Very fast | Standard |
| **Batch Processing** | ✅ Native vLLM | Manual batching |
| **Multi-GPU** | ✅ Tensor parallelism | Per model |
| **LoRA** | ✅ Built-in | Depends on model |
| **When to Use** | **Most cases** | OpenAI API, multiple benchmarks |

**Recommendation**: Use `vllm_infer.py` for all vLLM-compatible models. Only use `run_eval.py` when you need OpenAI API or multi-benchmark evaluation.

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or memory utilization
--batch_size 512 \
--gpu_memory_utilization 0.8
```

### Wrong Template
```bash
# Make sure template matches your model
--template qwen2_5_vl  # For Qwen2.5-VL
--template llama3      # For Llama-3
```

### Benchmark Not Found
```bash
# Check benchmark is registered in BENCHMARK_REGISTRY
# See embodied_eval/__init__.py
```

## Advanced Usage

### Custom vLLM Parameters
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_name robovqa \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.95 \
    --max_model_len 8192 \
    --swap_space 16
```

### Custom Generation Parameters
```bash
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_name robovqa \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 50 \
    --max_tokens 512 \
    --repetition_penalty 1.1
```

### Saving Predictions Only (No Evaluation)
The script always runs evaluation automatically. If you only want predictions:
```python
# Modify vllm_infer.py to skip evaluation
# Or extract predictions from generated_predictions.jsonl
```

## Integration with Benchmark Framework

### How vllm_infer.py Uses Benchmarks

```python
# 1. Load benchmark
from embodied_eval import BENCHMARK_REGISTRY
benchmark = BENCHMARK_REGISTRY.get("robovqa")(config)

# 2. Get samples
for sample in benchmark:
    # 3. Preprocess
    processed_sample = benchmark.preprocess(sample)
    
    # 4. Generate prompt
    messages = benchmark.generate_prompt(processed_sample, model=None)
    
    # 5. Convert to vLLM format (done by vllm_infer.py)
    # ...

# 6. Run vLLM inference
outputs = llm.generate(prompts, sampling_params)

# 7. Evaluate
results = benchmark.evaluate(predictions, ground_truths, metadata)
```

### Creating Custom Benchmarks for vLLM

Your benchmark must implement:
- `__iter__()`: Iterate over samples
- `preprocess(sample)`: Preprocess sample
- `generate_prompt(sample, model)`: Generate chat messages
- `evaluate(predictions, ground_truths, metadata)`: Compute metrics

Example:
```python
from embodied_eval.base import BaseBenchmark, Sample

class MyBenchmark(BaseBenchmark):
    def _load_data(self):
        # Load your data
        return samples
    
    def preprocess(self, sample):
        # Preprocess sample
        return sample
    
    def generate_prompt(self, sample, model):
        # Return list of message dicts
        return [{"role": "user", "content": sample.question}]
    
    def evaluate(self, predictions, ground_truths, metadata):
        # Compute metrics
        return {"accuracy": accuracy}
```

Then register it:
```python
from embodied_eval import BENCHMARK_REGISTRY
BENCHMARK_REGISTRY.register("my_benchmark", MyBenchmark)
```

## See Also

- [examples/using_vllm_infer.py](examples/using_vllm_infer.py) - Complete usage examples
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [README_NEW.md](README_NEW.md) - Full framework documentation
- [vLLM Documentation](https://docs.vllm.ai/) - Official vLLM docs

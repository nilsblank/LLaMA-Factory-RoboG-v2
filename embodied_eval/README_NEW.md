# Embodied Evaluation Framework - Updated Architecture

## Overview

This framework provides a flexible, configuration-driven system for evaluating embodied AI models on various benchmarks. It uses **vLLM** for efficient batch inference (RECOMMENDED) and **Hydra** for configuration management.

## Recommended Usage: vLLM Inference

**For most use cases, use `vllm_infer.py` directly** - it provides:
- ✅ Efficient batch processing with vLLM
- ✅ Multi-GPU tensor parallelism
- ✅ Automatic benchmark loading and evaluation
- ✅ Simple command-line interface

```bash
# Quick start - efficient vLLM inference
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_name robovqa \
    --batch_size 1024
```

See [VLLM_GUIDE.md](VLLM_GUIDE.md) for detailed vLLM usage.

## Alternative: Hydra Multi-Benchmark Runner

Use `run_eval.py` when you need to:
- Evaluate multiple benchmarks × multiple models
- Use non-vLLM models (OpenAI API, etc.)
- Customize evaluation pipelines

```bash
python -m embodied_eval.run_eval \
    'benchmarks=[robovqa,other]' \
    'models=[openai,llamafactory]'
```

## Key Features

### 1. **Merged Dataset + Evaluator → Benchmark**
- Single `BaseBenchmark` class handles both data loading and evaluation
- Simpler architecture, less boilerplate

### 2. **Efficient vLLM Integration**
- `vllm_infer.py` script adapted from LlamaFactory's vLLM implementation
- Loads benchmarks seamlessly while maintaining vLLM optimizations
- STANDARD way to run inference

### 3. **Restructured Config Directory**
```
configs/
├── config.yaml              # Top-level Hydra config
├── benchmarks/              # Benchmark configs
│   └── robovqa.yaml
└── models/                  # Model configs
    ├── mock.yaml
    ├── llamafactory.yaml
    └── openai.yaml
```

### 4. **Multi-Model Support**
- vLLM models: Use `vllm_infer.py` (RECOMMENDED)
- OpenAI API: GPT-4, GPT-4o, vision models
- LlamaFactory: Integration with existing training pipeline

## Architecture

### Core Classes

#### `BaseBenchmark`
Combines data loading and evaluation:
- `_load_data()`: Load dataset from disk
- `preprocess()`: Preprocess samples
- `generate_prompt()`: Create model prompts
- `evaluate()`: Compute metrics
- `print_summary()`: Display results

**Important for vLLM**: When using vLLM inference, model-specific processing (e.g., bbox normalization) should be done in `preprocess()` since vLLM bypasses BaseModel classes.

#### `BaseModel`
Model wrapper interface:
- `generate()`: Generate predictions
- `process_bbox()`: Handle bbox normalization
- `process_image()`: Preprocess images
- `parse_output()`: Parse model outputs

**Note**: Only used for non-vLLM models (OpenAI, LlamaFactory, etc.)

### Model-Specific Processing

#### For vLLM Models
Processing happens in **benchmark classes** (`BaseBenchmark` methods):
- `preprocess(sample)` - Normalize bboxes, preprocess data
- `generate_prompt(sample, model)` - Format prompts with processed data

Example:
```python
class RoboVQABenchmark(BaseBenchmark):
    def preprocess(self, sample):
        # Normalize bboxes for vLLM inference
        if 'bbox' in sample.metadata:
            bbox = sample.metadata['bbox']
            # Convert to normalized format
            sample.metadata['bbox_norm'] = normalize_bbox(bbox, ...)
        return sample
```

#### For Regular Models (OpenAI, LlamaFactory)
Processing happens in **model classes** (`BaseModel` methods):
- `process_bbox(bbox, ...)` - Model-specific bbox format
- `parse_output(output)` - Parse model outputs
- `process_image(image)` - Model-specific preprocessing

Example:
```python
class OpenAIModel(BaseModel):
    def process_bbox(self, bbox, original_size, target_size, format="xyxy"):
        # OpenAI-specific bbox processing
        return normalized_bbox
```

### Two Inference Paths

```
vLLM Path (FAST):
  Benchmark.preprocess() → Benchmark.generate_prompt() → vllm_infer() → Benchmark.evaluate()

Regular Model Path:
  Benchmark.preprocess() → Benchmark.generate_prompt() → Model.generate() → Model.parse_output() → Benchmark.evaluate()
```

### Configuration

All components use **OmegaConf/Hydra** for configuration:

```yaml
# configs/config.yaml
defaults:
  - benchmarks:
      - robovqa
  - models:
      - llamafactory
      - openai

output_dir: ./eval_results
save_predictions: true
verbose: true
```

```yaml
# configs/benchmarks/robovqa.yaml
name: robovqa
_target_: embodied_eval.robovqa_benchmark.RoboVQABenchmark

data_dir: /path/to/robovqa/tfrecords
split: validation
metrics:
  - bleu
  - rouge-l
```

```yaml
# configs/models/openai.yaml
name: openai_model
_target_: embodied_eval.models.OpenAIModel

model_path: gpt-4o
api_key: ${oc.env:OPENAI_API_KEY}
generation_kwargs:
  temperature: 0.7
  max_tokens: 512
```

## Usage

### Method 1: vLLM Inference (RECOMMENDED)

```bash
# Basic usage
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --template qwen2_5_vl \
    --benchmark_name robovqa \
    --batch_size 1024

# Use benchmark config file
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_config embodied_eval/configs/benchmarks/robovqa.yaml \
    --batch_size 1024

# Multi-GPU with LoRA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/base_model \
    --adapter_name_or_path /path/to/lora \
    --benchmark_name robovqa \
    --batch_size 2048
```

See [VLLM_GUIDE.md](VLLM_GUIDE.md) and `examples/using_vllm_infer.py` for more examples.

### Method 2: Hydra Config Runner

```bash
# Run evaluation with default config
python -m embodied_eval.run_eval

# Override config options
python -m embodied_eval.run_eval output_dir=./my_results

# Select different benchmarks/models
python -m embodied_eval.run_eval \
    'benchmarks=[robovqa]' \
    'models=[openai,llamafactory]'
```

### Programmatic Usage

```python
from embodied_eval.robovqa_benchmark import RoboVQABenchmark
from embodied_eval.models import MockModel

# Load benchmark
benchmark = RoboVQABenchmark("configs/benchmarks/robovqa.yaml")

# Load model
model = MockModel("configs/models/mock.yaml")

# Generate predictions
predictions = []
for sample in benchmark:
    prompt = benchmark.generate_prompt(sample, model)
    pred = model.generate(prompt, images=sample.images)
    predictions.append(pred)

# Evaluate
ground_truths = [s.answer for s in benchmark.samples]
metadata = [s.metadata for s in benchmark.samples]
results = benchmark.evaluate(predictions, ground_truths, metadata)

# Print results
benchmark.print_summary()
```

## Creating Custom Benchmarks

```python
from embodied_eval.base import BaseBenchmark, BaseModel, Sample

class MyBenchmark(BaseBenchmark):
    def _validate_config(self):
        # Check required config keys
        if 'data_dir' not in self.config:
            raise ValueError("data_dir required")
    
    def _load_data(self):
        # Load your dataset
        for item in load_my_data(self.data_dir):
            sample = Sample(
                question=item['question'],
                answer=item['answer'],
                images=item.get('images'),
                metadata={'task': item['task']}
            )
            self.samples.append(sample)
    
    def preprocess(self, sample: Sample) -> Sample:
        # Optional preprocessing
        return sample
    
    def generate_prompt(self, sample: Sample, model: BaseModel) -> str:
        # Format prompt for model
        return f"Question: {sample.question}\nAnswer:"
    
    def evaluate(self, predictions, ground_truths, metadata):
        # Compute metrics
        accuracy = sum(p == gt for p, gt in zip(predictions, ground_truths)) / len(predictions)
        return {'accuracy': accuracy}
```

Then register and use:

```python
# In run_eval.py or your script
from embodied_eval.run_eval import register_benchmark
register_benchmark("mybenchmark")(MyBenchmark)
```

## Creating Custom Models

```python
from embodied_eval.base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Load your model
        self.model = load_my_model(self.model_path)
    
    def generate(self, prompt, images=None, **kwargs):
        # Run inference
        output = self.model.predict(prompt, images)
        return output
    
    def process_bbox(self, bbox, original_size, target_size, format="xyxy"):
        # Custom bbox processing if needed
        return super().process_bbox(bbox, original_size, target_size, format)
```

## Supported Models

### MockModel
For testing only. Echoes prompts or returns fixed responses.

```yaml
# configs/models/mock.yaml
name: mock_model
_target_: embodied_eval.models.MockModel
response_mode: echo  # or 'fixed', 'random'
```

### LlamaFactoryModel
Template for LlamaFactory integration.

```yaml
# configs/models/llamafactory.yaml
name: llamafactory_model
_target_: embodied_eval.models.LlamaFactoryModel
model_path: /path/to/checkpoint
use_smart_resize: true  # For Qwen2.5-VL
```

### VLLMModel (Recommended for Batch Inference)
Efficient batch inference using vLLM with tensor/pipeline parallelism.

```yaml
# configs/models/vllm.yaml
name: vllm_model
_target_: embodied_eval.vllm_model.VLLMModel
model_path: /path/to/checkpoint
template: default
cutoff_len: 2048
pipeline_parallel_size: 1
generation_kwargs:
  temperature: 0.95
  max_new_tokens: 1024
```

**Usage:**
```bash
# Run with vLLM (efficient batch inference)
python -m embodied_eval.run_eval \
    --config-name config_vllm \
    models.model_path=/path/to/checkpoint \
    batch_size=1024
```

**Features:**
- Tensor parallelism for multi-GPU inference
- Pipeline parallelism
- LoRA adapter support
- Multimodal inputs (images, videos, audio)
- Efficient batch generation

### OpenAIModel
OpenAI API integration (GPT-4, GPT-4o, etc.).

```yaml
# configs/models/openai.yaml
name: openai_model
_target_: embodied_eval.models.OpenAIModel
model_path: gpt-4o
api_key: ${oc.env:OPENAI_API_KEY}
```

## Supported Benchmarks

### RoboVQA
Visual question answering for robotics tasks.

```yaml
# configs/benchmarks/robovqa.yaml
name: robovqa
_target_: embodied_eval.robovqa_benchmark.RoboVQABenchmark
data_dir: /path/to/robovqa/tfrecords
split: validation  # train, validation, or test
metrics:
  - bleu
  - rouge-l
max_samples: 100  # Optional: limit for quick testing
```

**Metrics:**
- BLEU (sacrebleu)
- ROUGE-L (rouge-score)
- Task-type-specific metrics (binary, discrete, open)

## Output Structure

```
eval_results/
├── robovqa_mock_predictions.jsonl      # Predictions in JSONL format
├── robovqa_mock_results.json           # Evaluation metrics
├── robovqa_openai_predictions.jsonl
├── robovqa_openai_results.json
└── all_results.json                     # Combined results for all pairs
```

### Prediction File Format (JSONL)
```json
{"prediction": "...", "ground_truth": "...", "metadata": {...}}
{"prediction": "...", "ground_truth": "...", "metadata": {...}}
```

### Results File Format (JSON)
```json
{
  "overall": {
    "bleu": 45.2,
    "rouge_l": 0.523
  },
  "binary": {
    "bleu": 52.1,
    "rouge_l": 0.601,
    "num_samples": 150
  },
  ...
}
```

## Dependencies

```bash
pip install hydra-core omegaconf
pip install tensorflow  # For RoboVQA TFRecord loading
pip install sacrebleu rouge-score  # For metrics
pip install openai  # For OpenAI models (optional)
pip install pillow numpy tqdm  # General utilities
```

## Migration from Old Framework

If you have code using the old `BaseDataset` + `BaseEvaluator` structure:

1. **Merge your dataset and evaluator into a single benchmark class**
   - Inherit from `BaseBenchmark` instead of `BaseDataset`
   - Move `evaluate()` method from evaluator into benchmark
   
2. **Update config initialization**
   - Old: `__init__(self, config_path=None, config_dict=None)`
   - New: `__init__(self, cfg: Union[str, Path, DictConfig, Dict])`

3. **Move configs to new directory structure**
   - Dataset configs → `configs/benchmarks/`
   - Model configs → `configs/models/`
   - Add `_target_` field pointing to class

4. **Update imports**
   - Old: `from embodied_eval.base import BaseDataset, BaseEvaluator`
   - New: `from embodied_eval.base import BaseBenchmark`

## Advanced Features

### Environment Variable Interpolation
```yaml
api_key: ${oc.env:OPENAI_API_KEY}  # Read from environment
data_dir: ${oc.env:DATA_ROOT}/robovqa
```

### Config Composition
```bash
# Mix and match configs
python -m embodied_eval.run_eval \
    +benchmarks@benchmarks.my_bench=custom_benchmark \
    +models@models.my_model=custom_model
```

### Hydra Multirun
```bash
# Run multiple experiments
python -m embodied_eval.run_eval -m \
    models=mock,llamafactory,openai \
    benchmarks=robovqa
```

## Troubleshooting

**Import errors:**
```python
# Make sure to run from repo root
cd /path/to/LLaMA-Factory-RoboG-v2
python -m embodied_eval.run_eval
```

**Config not found:**
```python
# Hydra looks for configs/ relative to run_eval.py
# Ensure configs/ is in embodied_eval/configs/
```

**OpenAI API errors:**
```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Or specify in config
python -m embodied_eval.run_eval models.api_key=sk-...
```

## Examples

See `embodied_eval/examples/` for:
- `getting_started.py`: Basic usage
- `custom_benchmark.py`: Creating custom benchmarks
- `batch_evaluation.py`: Evaluating multiple benchmarks

## License

Apache 2.0 - See LICENSE file

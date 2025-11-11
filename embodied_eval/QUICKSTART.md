# Quick Start Guide - Embodied Evaluation Framework v2

## Installation

```bash
# Install dependencies
pip install hydra-core omegaconf
pip install tensorflow  # For RoboVQA
pip install sacrebleu rouge-score  # For metrics
pip install vllm  # For efficient inference (RECOMMENDED)
pip install openai  # For OpenAI models (optional)
pip install pillow numpy tqdm
```

## Basic Usage - vLLM Inference (RECOMMENDED)

### 1. Quick Start with vLLM
```bash
# This is the STANDARD way - uses efficient vLLM batch inference
cd /path/to/LLaMA-Factory-RoboG-v2

python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/your/model \
    --template qwen2_5_vl \
    --benchmark_name robovqa \
    --benchmark_data_dir /path/to/robovqa/tfrecords \
    --batch_size 1024
```

### 2. Using Benchmark Config File
```bash
# Use a pre-configured benchmark
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_config embodied_eval/configs/benchmarks/robovqa.yaml \
    --batch_size 1024
```

### 3. Multi-GPU Inference
```bash
# Automatically uses all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/70B/model \
    --benchmark_name robovqa \
    --batch_size 2048
```

### 4. Quick Test (10 samples)
```bash
# Test with small sample
python -m embodied_eval.vllm_infer \
    --model_name_or_path /path/to/model \
    --benchmark_name robovqa \
    --max_samples 10 \
    --batch_size 10
```

## Alternative - Hydra Config (For Multiple Benchmarks/Models)

### 1. Run with Default Config
```bash
python -m embodied_eval.run_eval
```

### 2. Override Options
```bash
python -m embodied_eval.run_eval \
    output_dir=./my_results \
    benchmarks.max_samples=100
```

## Programmatic Usage

### Simple Example
```python
from embodied_eval.robovqa_benchmark import RoboVQABenchmark
from embodied_eval.models import MockModel

# Load benchmark
benchmark = RoboVQABenchmark({
    "name": "robovqa",
    "data_dir": "/path/to/robovqa/tfrecords",
    "split": "validation",
    "metrics": ["bleu", "rouge-l"],
    "max_samples": 100
})

# Load model
model = MockModel({
    "name": "mock",
    "response_mode": "echo"
})

# Generate and evaluate
predictions = []
for sample in benchmark:
    prompt = benchmark.generate_prompt(sample, model)
    pred = model.generate(prompt, images=sample.images)
    predictions.append(pred)

ground_truths = [s.answer for s in benchmark]
results = benchmark.evaluate(predictions, ground_truths)
benchmark.print_summary()
```

### Using Config Files
```python
# Load from YAML
benchmark = RoboVQABenchmark("embodied_eval/configs/benchmarks/robovqa.yaml")
model = MockModel("embodied_eval/configs/models/mock.yaml")

# Rest is the same...
```

## Creating Custom Benchmarks

### 1. Define Benchmark Class
```python
# my_benchmark.py
from embodied_eval.base import BaseBenchmark, Sample

class MyBenchmark(BaseBenchmark):
    def _validate_config(self):
        if 'data_file' not in self.config:
            raise ValueError("data_file required")
    
    def _load_data(self):
        import json
        with open(self.data_dir / self.config.data_file) as f:
            data = json.load(f)
        
        for item in data:
            sample = Sample(
                question=item['question'],
                answer=item['answer'],
                metadata={'id': item['id']}
            )
            self.samples.append(sample)
    
    def preprocess(self, sample):
        # Optional preprocessing
        return sample
    
    def generate_prompt(self, sample, model):
        return f"Q: {sample.question}\nA:"
    
    def evaluate(self, predictions, ground_truths, metadata=None):
        # Simple accuracy
        correct = sum(p.strip() == gt.strip() 
                     for p, gt in zip(predictions, ground_truths))
        return {
            'overall': {
                'accuracy': correct / len(predictions),
                'num_samples': len(predictions)
            }
        }
```

### 2. Create Config File
```yaml
# configs/benchmarks/mybenchmark.yaml
name: mybenchmark
_target_: embodied_eval.my_benchmark.MyBenchmark

data_dir: /path/to/data
data_file: questions.json
```

### 3. Register and Use
```python
# In run_eval.py, add:
from . import my_benchmark
register_benchmark("mybenchmark")(my_benchmark.MyBenchmark)

# Then use:
# python -m embodied_eval.run_eval benchmarks=mybenchmark
```

## Creating Custom Models

### 1. Define Model Class
```python
# my_model.py
from embodied_eval.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Load your model
        self.model = self._load_model()
    
    def _load_model(self):
        # Your model loading logic
        pass
    
    def generate(self, prompt, images=None, **kwargs):
        # Handle chat format
        if isinstance(prompt, dict):
            prompt = prompt['content']
        
        # Generate with your model
        output = self.model.predict(prompt)
        return output
```

### 2. Create Config File
```yaml
# configs/models/mymodel.yaml
name: mymodel
_target_: embodied_eval.my_model.MyModel

model_path: /path/to/checkpoint
generation_kwargs:
  temperature: 0.7
  max_tokens: 512
```

### 3. Register and Use
```python
# In run_eval.py, add:
from . import my_model
register_model("mymodel")(my_model.MyModel)

# Then use:
# python -m embodied_eval.run_eval models=mymodel
```

## Using OpenAI Models

### 1. Set API Key
```bash
export OPENAI_API_KEY=sk-...
```

### 2. Run Evaluation
```bash
python -m embodied_eval.run_eval models=openai
```

### 3. Programmatic Usage
```python
from embodied_eval.models import OpenAIModel

model = OpenAIModel({
    "name": "gpt4",
    "model_path": "gpt-4o",
    "api_key": "sk-...",  # Or use env var
    "generation_kwargs": {
        "temperature": 0.7,
        "max_tokens": 512
    }
})

# Works with text and images
response = model.generate(
    prompt="What's in this image?",
    images=[image_array]
)
```

## Configuration Tips

### Environment Variables
```yaml
# configs/models/openai.yaml
api_key: ${oc.env:OPENAI_API_KEY}
data_dir: ${oc.env:DATA_ROOT}/robovqa
```

### Conditional Config
```yaml
# Different configs for different models
generation_kwargs:
  temperature: ${oc.select:temperature,0.7}
  max_tokens: ${oc.select:max_tokens,512}
```

### Config Composition
```yaml
# configs/config.yaml
defaults:
  - benchmarks:
      - robovqa
  - models:
      - _self_  # Include this file's config
      - llamafactory

# Override from command line:
# python -m embodied_eval.run_eval +models=openai
```

## Common Workflows

### Quick Test Run
```bash
# Test with 10 samples
python -m embodied_eval.run_eval \
    benchmarks.max_samples=10 \
    models=mock
```

### Full Evaluation
```bash
# Evaluate all benchmarks with all models
python -m embodied_eval.run_eval \
    'benchmarks=[robovqa,bench2]' \
    'models=[llamafactory,openai]' \
    output_dir=./results_full
```

### Debug Mode
```bash
# Verbose output, save predictions
python -m embodied_eval.run_eval \
    verbose=true \
    save_predictions=true \
    benchmarks.max_samples=5
```

### Compare Models
```bash
# Evaluate same benchmark with different models
python -m embodied_eval.run_eval \
    benchmarks=robovqa \
    'models=[mock,llamafactory,openai]'

# Results in:
# - robovqa_mock_results.json
# - robovqa_llamafactory_results.json
# - robovqa_openai_results.json
# - all_results.json (combined)
```

## Output Structure

```
eval_results/
├── robovqa_mock_predictions.jsonl
├── robovqa_mock_results.json
├── robovqa_openai_predictions.jsonl
├── robovqa_openai_results.json
├── all_results.json
└── hydra_runs/
    └── 2025-01-20_14-30-00/
        └── .hydra/
            ├── config.yaml
            └── overrides.yaml
```

## Troubleshooting

### Import Errors
```bash
# Always run from repo root
cd /path/to/LLaMA-Factory-RoboG-v2
python -m embodied_eval.run_eval  # Not: python embodied_eval/run_eval.py
```

### Config Not Found
```python
# Hydra looks for configs/ relative to run_eval.py
# Make sure structure is:
# embodied_eval/
#   ├── run_eval.py
#   └── configs/
#       ├── config.yaml
#       ├── benchmarks/
#       └── models/
```

### Module Registration
```python
# Make sure to register in run_eval.py:
from . import my_benchmark
register_benchmark("mybench")(my_benchmark.MyBenchmark)
```

### OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Or pass directly
python -m embodied_eval.run_eval \
    models=openai \
    models.api_key=sk-...
```

## Best Practices

### 1. Use Configs for Everything
```python
# Good ✅
benchmark = MyBenchmark("configs/benchmarks/mybench.yaml")

# Avoid ❌
benchmark = MyBenchmark({"data_dir": "/path", "split": "val", ...})
```

### 2. Test with Small Samples First
```bash
# Always test with max_samples first
python -m embodied_eval.run_eval benchmarks.max_samples=10
```

### 3. Save Predictions
```yaml
# In config.yaml
save_predictions: true  # Enables debugging and re-evaluation
```

### 4. Use Descriptive Names
```yaml
# Good ✅
name: robovqa_validation_full
name: gpt4o_temp0.7

# Avoid ❌
name: dataset1
name: model2
```

### 5. Version Your Configs
```bash
# Keep configs under version control
git add configs/benchmarks/mybenchmark_v1.yaml
git commit -m "Add MyBenchmark v1 config"
```

## Advanced Usage

### Hydra Multirun
```bash
# Run multiple experiments
python -m embodied_eval.run_eval -m \
    models=mock,openai \
    benchmarks.split=train,validation,test
```

### Custom Hydra Config Path
```python
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # ...
```

### Programmatic Hydra
```python
from hydra import compose, initialize

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="config", overrides=[
        "benchmarks=robovqa",
        "models=mock"
    ])
    # Use cfg...
```

## What's Next?

1. **Try the examples:**
   ```bash
   python embodied_eval/examples/getting_started_new.py
   ```

2. **Read full docs:**
   - `README_NEW.md` - Complete documentation
   - `MIGRATION.md` - Migration from old framework
   - `RESTRUCTURING_SUMMARY.md` - What changed and why

3. **Create your benchmarks and models:**
   - See examples above
   - Check `robovqa_benchmark.py` for reference

4. **Run comprehensive evaluations:**
   ```bash
   python -m embodied_eval.run_eval \
       'benchmarks=[bench1,bench2,bench3]' \
       'models=[model1,model2,model3]'
   ```

Happy evaluating! 🚀

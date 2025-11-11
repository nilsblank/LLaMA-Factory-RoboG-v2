# Migration Guide: Old Framework → New Benchmark-Based Framework

## Overview of Changes

The framework has been restructured to:
1. **Merge Dataset + Evaluator** into a single `BaseBenchmark` class
2. **Reorganize configs** into `benchmarks/` and `models/` subdirectories
3. **Support multi-evaluation** with Hydra defaults list pattern
4. **Add OpenAI API support** for commercial model evaluation

## Step-by-Step Migration

### 1. Update Base Class

**Old Code:**
```python
from embodied_eval.base import BaseDataset, BaseEvaluator

class MyDataset(BaseDataset):
    def _validate_config(self):
        # ...
    
    def _load_data(self):
        # ...
    
    def preprocess(self, sample):
        # ...
    
    def generate_prompt(self, sample, model):
        # ...

class MyEvaluator(BaseEvaluator):
    def evaluate(self, predictions, ground_truths, model=None):
        # ...
```

**New Code:**
```python
from embodied_eval.base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    # Merge dataset and evaluator functionality
    
    def _validate_config(self):
        # Same as before
        pass
    
    def _load_data(self):
        # Same as before
        pass
    
    def preprocess(self, sample):
        # Same as before
        pass
    
    def generate_prompt(self, sample, model):
        # Same as before - but can return dict for chat format
        pass
    
    def evaluate(self, predictions, ground_truths, metadata=None):
        # Moved from evaluator class
        # Compute your metrics here
        pass
```

### 2. Update Constructor

**Old Code:**
```python
def __init__(self, config_path=None, config_dict=None):
    if config_path is not None:
        self.config = OmegaConf.load(config_path)
    elif config_dict is not None:
        self.config = OmegaConf.create(config_dict)
    else:
        raise ValueError("...")
    
    # Custom initialization
```

**New Code:**
```python
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
    # Parent class handles config loading
    super().__init__(cfg)
    
    # Your custom initialization (optional)
    # self.config is already set by parent
```

### 3. Reorganize Config Files

**Old Structure:**
```
configs/
├── robovqa_example.yaml
├── robovqa_train.yaml
├── mock_model.yaml
└── llamafactory_model.yaml
```

**New Structure:**
```
configs/
├── config.yaml              # Top-level Hydra config
├── benchmarks/
│   ├── robovqa.yaml
│   └── another_benchmark.yaml
└── models/
    ├── mock.yaml
    ├── llamafactory.yaml
    └── openai.yaml
```

### 4. Update Config Format

Add `_target_` field pointing to your class:

**Old Config (benchmarks/robovqa.yaml):**
```yaml
data_dir: /path/to/data
split: validation
generation_kwargs:
  temperature: 0.7
```

**New Config (benchmarks/robovqa.yaml):**
```yaml
name: robovqa
_target_: embodied_eval.robovqa_benchmark.RoboVQABenchmark

data_dir: /path/to/data
split: validation

# Note: generation_kwargs moved to model config
```

**Old Config (models/mock.yaml):**
```yaml
model_name: mock
model_path: ""
```

**New Config (models/mock.yaml):**
```yaml
name: mock_model
_target_: embodied_eval.models.MockModel

response_mode: echo
generation_kwargs:
  temperature: 0.7
  max_tokens: 256
```

### 5. Create Top-Level Config

Create `configs/config.yaml`:

```yaml
defaults:
  - benchmarks:
      - robovqa
  - models:
      - mock

output_dir: ./eval_results
save_predictions: true
verbose: true

hydra:
  run:
    dir: ${output_dir}/hydra_runs/${now:%Y-%m-%d_%H-%M-%S}
```

### 6. Update Evaluation Script

**Old Code:**
```python
from embodied_eval.robovqa_dataset import RoboVQADataset
from embodied_eval.robovqa_evaluator import RoboVQAEvaluator
from embodied_eval.models import MockModel

# Load components
dataset = RoboVQADataset(config_path="configs/robovqa.yaml")
model = MockModel(config_path="configs/mock.yaml")
evaluator = RoboVQAEvaluator()

# Generate predictions
predictions = []
for sample in dataset:
    prompt = dataset.generate_prompt(sample, model)
    pred = model.generate(prompt)
    predictions.append(pred)

# Evaluate
ground_truths = [s.answer for s in dataset.samples]
results = evaluator.evaluate(predictions, ground_truths, model)
```

**New Code (Option 1: Use run_eval.py):**
```bash
# Just use Hydra runner
python -m embodied_eval.run_eval
```

**New Code (Option 2: Programmatic):**
```python
from embodied_eval.robovqa_benchmark import RoboVQABenchmark
from embodied_eval.models import MockModel

# Load benchmark (dataset + evaluator merged)
benchmark = RoboVQABenchmark("configs/benchmarks/robovqa.yaml")
model = MockModel("configs/models/mock.yaml")

# Generate predictions
predictions = []
for sample in benchmark:
    prompt = benchmark.generate_prompt(sample, model)
    pred = model.generate(prompt, images=sample.images)
    predictions.append(pred)

# Evaluate (now part of benchmark)
ground_truths = [s.answer for s in benchmark.samples]
metadata = [s.metadata for s in benchmark.samples]
results = benchmark.evaluate(predictions, ground_truths, metadata)

# Print results (also part of benchmark)
benchmark.print_summary()
```

### 7. Update Model Class

**Old Code:**
```python
class MyModel(BaseModel):
    def __init__(self, config_path=None, config_dict=None):
        super().__init__(config_path, config_dict)
        # Load model
    
    def generate(self, prompt: str, images=None, **kwargs):
        # Generate
        pass
```

**New Code:**
```python
from typing import Union, Dict
from pathlib import Path
from omegaconf import DictConfig

class MyModel(BaseModel):
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        super().__init__(cfg)
        # self.config, self.name, self.generation_kwargs set by parent
        # Load model
    
    def generate(
        self, 
        prompt: Union[str, Dict[str, str]],  # Now supports chat format!
        images=None, 
        videos=None,  # New multimodal support
        audios=None,
        **kwargs
    ):
        # Handle chat format if needed
        if isinstance(prompt, dict):
            prompt_text = prompt.get('content', '')
        else:
            prompt_text = prompt
        
        # Generate
        pass
```

### 8. Register Components

Add registration in `run_eval.py`:

```python
from embodied_eval.run_eval import register_benchmark, register_model
from embodied_eval.my_benchmark import MyBenchmark
from embodied_eval.my_model import MyModel

# Register so Hydra can find them
register_benchmark("mybenchmark")(MyBenchmark)
register_model("mymodel")(MyModel)
```

Or add to the imports section:
```python
# In run_eval.py
from . import robovqa_benchmark, models, my_benchmark, my_model

register_benchmark("robovqa")(robovqa_benchmark.RoboVQABenchmark)
register_benchmark("mybenchmark")(my_benchmark.MyBenchmark)

register_model("mock")(models.MockModel)
register_model("mymodel")(my_model.MyModel)
```

## Breaking Changes

### 1. No More Separate Evaluator
- **Old:** `BaseDataset` + `BaseEvaluator` as separate classes
- **New:** `BaseBenchmark` combines both
- **Action:** Merge your evaluator's `evaluate()` method into your dataset class

### 2. Config Constructor Signature
- **Old:** `__init__(self, config_path=None, config_dict=None)`
- **New:** `__init__(self, cfg: Union[str, Path, DictConfig, Dict])`
- **Action:** Update signature and call `super().__init__(cfg)`

### 3. Config Directory Structure
- **Old:** Flat `configs/` directory
- **New:** Organized into `configs/benchmarks/` and `configs/models/`
- **Action:** Move and rename config files, add `_target_` field

### 4. Model Generate Signature
- **Old:** `generate(prompt: str, images=None, **kwargs)`
- **New:** `generate(prompt: Union[str, Dict], images=None, videos=None, audios=None, **kwargs)`
- **Action:** Add support for chat format and new modalities

### 5. Evaluation Results Storage
- **Old:** Evaluator stores results in `self.results`
- **New:** Benchmark stores results in `self.results`
- **Action:** Access results via benchmark instance

## New Features You Can Use

### 1. Multi-Benchmark Multi-Model Evaluation
```bash
python -m embodied_eval.run_eval \
    benchmarks=robovqa,another_bench \
    models=mock,llamafactory,openai
```

### 2. OpenAI API Integration
```python
from embodied_eval.models import OpenAIModel

model = OpenAIModel({
    "name": "gpt4",
    "model_path": "gpt-4o",
    "api_key": "sk-...",
    "generation_kwargs": {"temperature": 0.7}
})

# Automatically handles chat format and vision inputs
response = model.generate(prompt, images=[img_array])
```

### 3. Chat Format Support
```python
# Models can now handle OpenAI-style chat format
prompt_dict = {
    "role": "user",
    "content": "What is in this image?"
}
response = model.generate(prompt_dict, images=[img])
```

### 4. Environment Variable Interpolation
```yaml
# configs/models/openai.yaml
api_key: ${oc.env:OPENAI_API_KEY}
data_dir: ${oc.env:DATA_ROOT}/robovqa
```

### 5. Hydra Multirun
```bash
# Run multiple experiments in parallel
python -m embodied_eval.run_eval -m \
    models=mock,llamafactory \
    benchmarks=robovqa \
    benchmarks.max_samples=10,100,1000
```

## Testing Your Migration

1. **Test benchmark loading:**
```python
benchmark = YourBenchmark("configs/benchmarks/your_benchmark.yaml")
assert len(benchmark) > 0
assert hasattr(benchmark, 'evaluate')
```

2. **Test model loading:**
```python
model = YourModel("configs/models/your_model.yaml")
assert hasattr(model, 'generate')
```

3. **Test evaluation:**
```python
predictions = ["pred1", "pred2"]
ground_truths = ["gt1", "gt2"]
results = benchmark.evaluate(predictions, ground_truths)
assert 'overall' in results or len(results) > 0
```

4. **Test Hydra integration:**
```bash
python -m embodied_eval.run_eval \
    benchmarks=your_benchmark \
    models=your_model \
    output_dir=./test_output \
    benchmarks.max_samples=2
```

## Common Issues

### Issue: "Config must have _target_ key"
**Solution:** Add `_target_` field to your config pointing to the class:
```yaml
_target_: embodied_eval.my_module.MyClass
```

### Issue: "Module not found"
**Solution:** Make sure to:
1. Import your module in `run_eval.py`
2. Register it with `register_benchmark()` or `register_model()`
3. Run from repo root: `python -m embodied_eval.run_eval`

### Issue: "BaseEvaluator not found"
**Solution:** `BaseEvaluator` has been removed. Use `BaseBenchmark` instead and merge evaluation logic.

### Issue: Old evaluation script doesn't work
**Solution:** Either:
1. Use new `run_eval.py` with Hydra
2. Update your script to use `BaseBenchmark` directly (see examples above)

## Backward Compatibility

The old `robovqa_dataset.py`, `robovqa_evaluator.py`, and original `run_eval.py` have been preserved as:
- `robovqa_dataset.py` (unchanged)
- `robovqa_evaluator.py` (unchanged)
- `run_eval_old.py` (backup of old runner)

You can continue using the old structure, but we recommend migrating to the new framework for:
- Cleaner architecture
- Multi-benchmark support
- OpenAI integration
- Better config management

## Need Help?

Check the examples:
- `examples/getting_started_new.py` - Basic usage
- `embodied_eval/robovqa_benchmark.py` - Complete benchmark implementation
- `embodied_eval/models.py` - Model implementations

Or refer to:
- `README_NEW.md` - Complete documentation
- `ARCHITECTURE.txt` - Design rationale

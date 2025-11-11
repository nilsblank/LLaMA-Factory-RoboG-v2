# Embodied AI Evaluation Framework

## Overview

I've created a complete, modular evaluation framework for embodied AI datasets based on your requirements. The framework is designed to be:

- **Config-driven**: All components instantiated via YAML files (using OmegaConf/Hydra)
- **Modular**: Easy to extend with new datasets, models, and evaluators
- **Model-agnostic**: Supports different model architectures with custom preprocessing
- **Production-ready**: Includes tests, documentation, and examples

## What's Included

### ✅ Core Requirements Met

1. **Dataset Classes** with YAML configuration
   - Abstract `BaseDataset` class
   - Full RoboVQA implementation
   - Parsing and preprocessing functionality
   - Prompt generation per model

2. **Model Classes** with preprocessing support
   - Abstract `BaseModel` class
   - Bounding box processing (e.g., for Qwen2.5-VL smart_resize)
   - Image preprocessing hooks
   - Output parsing

3. **Evaluator Classes** 
   - Abstract `BaseEvaluator` class
   - RoboVQA evaluator with BLEU/ROUGE metrics
   - Can leverage model-specific processing
   - Result saving and loading

4. **RoboVQA Implementation**
   - Based on google-deepmind/robovqa repository
   - Loads TFRecord format data
   - Parses task-based Q&A pairs
   - BLEU and ROUGE-L evaluation
   - Per-task-type metrics

## Quick Start

```bash
# 1. Install dependencies
cd /home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2
pip install -r embodied_eval/requirements.txt

# 2. Run tests to verify installation
python embodied_eval/test_framework.py

# 3. Try the examples
python embodied_eval/examples/getting_started.py
python embodied_eval/examples/custom_dataset.py

# 4. Run RoboVQA evaluation (requires data access)
python embodied_eval/run_eval.py embodied_eval/configs/robovqa_example.yaml
```

## File Structure

```
embodied_eval/
├── __init__.py                    # Package exports
├── base.py                        # Base classes (Dataset, Model, Evaluator)
├── robovqa_dataset.py             # RoboVQA dataset implementation
├── robovqa_evaluator.py           # RoboVQA BLEU/ROUGE evaluator  
├── models.py                      # Model wrappers (Mock, LlamaFactory)
├── utils.py                       # Utility functions (bbox, image, etc.)
├── run_eval.py                    # Main evaluation runner
├── test_framework.py              # Comprehensive test suite
│
├── configs/                       # Example configurations
│   ├── robovqa_example.yaml       # RoboVQA validation split
│   └── robovqa_train.yaml         # RoboVQA training split
│
├── examples/                      # Usage examples
│   ├── getting_started.py         # Basic usage demo
│   └── custom_dataset.py          # Custom dataset/evaluator example
│
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick reference guide
├── SUMMARY.md                     # Implementation summary
└── requirements.txt               # Python dependencies
```

## Usage Examples

### 1. Using Existing Components

```python
from embodied_eval import RoboVQADataset, RoboVQAEvaluator, MockModel

# Load dataset
dataset = RoboVQADataset(config_path='embodied_eval/configs/robovqa_example.yaml')

# Initialize model
model = MockModel(config_dict={'model_name': 'test'})

# Run inference
predictions = []
ground_truths = []
for sample in dataset:
    prompt = dataset.generate_prompt(sample, model)
    pred = model.generate(prompt, images=sample.images)
    predictions.append(pred)
    ground_truths.append(sample.answer)

# Evaluate
evaluator = RoboVQAEvaluator(use_bleu=True, use_rouge=True)
results = evaluator.evaluate(predictions, ground_truths)
evaluator.print_summary()
```

### 2. Creating Custom Components

```python
from embodied_eval.base import BaseDataset, BaseEvaluator, Sample

class MyDataset(BaseDataset):
    def _validate_config(self):
        # Validate required params
        pass
    
    def _load_data(self):
        # Load your data
        for item in my_data:
            sample = Sample(question=item['q'], answer=item['a'])
            self.samples.append(sample)
    
    def preprocess(self, sample):
        return sample
    
    def generate_prompt(self, sample, model):
        return f"Question: {sample.question}\\nAnswer:"

class MyEvaluator(BaseEvaluator):
    def evaluate(self, predictions, ground_truths, model=None):
        accuracy = sum(p == gt for p, gt in zip(predictions, ground_truths)) / len(predictions)
        return {'accuracy': accuracy}
```

### 3. Config-Driven Evaluation

```yaml
# my_config.yaml
dataset:
  name: mydataset
  data_dir: /path/to/data
  max_samples: 100
  generation_kwargs:
    temperature: 0.7
    max_tokens: 512

model:
  name: llamafactory
  checkpoint_path: /path/to/checkpoint

evaluator:
  name: myevaluator
  
output_dir: ./results
```

```bash
python embodied_eval/run_eval.py my_config.yaml
```

## Key Design Decisions

### 1. Config-Based Instantiation (OmegaConf)
All components accept either a config file path or a config dictionary:
```python
dataset = MyDataset(config_path='config.yaml')
# or
dataset = MyDataset(config_dict={'param': 'value'})
```

### 2. Sample Dataclass
Standardized multimodal sample format:
```python
Sample(
    question: str,
    answer: str,
    images: List[np.ndarray] = None,
    videos: List[np.ndarray] = None,
    metadata: Dict[str, Any] = None
)
```

### 3. Model-Specific Processing
Models provide hooks for preprocessing:
```python
model.process_bbox(bbox, width, height, format)  # Bbox normalization
model.process_image(image)                        # Image preprocessing
model.parse_output(output, format)                # Output parsing
```

### 4. Registry Pattern
Easy component registration:
```python
DATASET_REGISTRY['mydataset'] = MyDataset
EVALUATOR_REGISTRY['myeval'] = MyEvaluator
```

## RoboVQA Implementation Details

### Dataset
- Loads TFRecord files from Google Cloud Storage (or local)
- Parses XML-style tags (`<task:*>`, `<PRED>`, etc.)
- Extracts Q&A pairs using `Task` and `Tasks` classes
- Supports train/val splits

### Evaluator
- Computes sentence-level BLEU scores (sacrebleu)
- Computes ROUGE-L F1 scores
- Provides per-task-type breakdowns
- Matches reference implementation from google-deepmind/robovqa

### Task Parsing
Following the RoboVQA format:
```python
task = Task("<task:navigation> Q: Where to go? A: Kitchen")
qa_pairs = task.get_splits('A:')  # [("Q: Where to go?", "Kitchen")]
```

## Extending the Framework

### Adding a New Dataset

1. Create a class extending `BaseDataset`
2. Implement required methods
3. Create a YAML config
4. Register in `run_eval.py`

See `embodied_eval/examples/custom_dataset.py` for a complete example.

### Adding a New Evaluator

1. Create a class extending `BaseEvaluator`
2. Implement `evaluate()` method
3. Register in `run_eval.py`

### Adding Model Integration

1. Create a class extending `BaseModel`
2. Implement `generate()` method
3. Override preprocessing methods as needed
4. Register in `run_eval.py`

## Documentation

- **`README.md`**: Complete framework documentation
- **`QUICKSTART.md`**: Quick reference guide
- **`SUMMARY.md`**: Implementation summary
- **`examples/`**: Working code examples

## Testing

Run the test suite:
```bash
python embodied_eval/test_framework.py
```

Tests cover:
- Sample creation
- MockModel functionality
- RoboVQAEvaluator
- Bounding box processing
- Task parsing

## Dependencies

Core:
- `omegaconf`: Config management
- `numpy`: Array operations
- `tensorflow`: TFRecord loading (RoboVQA)
- `sacrebleu`: BLEU metrics
- `rouge-score`: ROUGE metrics
- `pillow`: Image operations
- `tqdm`: Progress bars

Optional:
- `sentence-transformers`: Semantic similarity
- `joblib`: Parallel processing

## Next Steps

1. **Test the framework**: Run `test_framework.py`
2. **Try examples**: Run scripts in `examples/`
3. **Integrate your model**: Create a model wrapper extending `BaseModel`
4. **Add your datasets**: Create dataset classes extending `BaseDataset`
5. **Customize evaluation**: Create evaluators extending `BaseEvaluator`

## Support

- Examples: `embodied_eval/examples/`
- Tests: `embodied_eval/test_framework.py`
- Full docs: `embodied_eval/README.md`
- Quick ref: `embodied_eval/QUICKSTART.md`

---

**Framework Version**: 0.1.0  
**Created**: November 2025  
**License**: Apache 2.0 (same as LlamaFactory)

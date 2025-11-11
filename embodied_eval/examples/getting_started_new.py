"""
Example: Getting started with the new Benchmark-based framework.

This example shows how to:
1. Load a benchmark from config
2. Load a model from config  
3. Run evaluation
4. Save and view results
"""

from pathlib import Path
from embodied_eval.robovqa_benchmark import RoboVQABenchmark
from embodied_eval.models import MockModel

def main():
    # 1. Create configs (can also load from YAML files)
    benchmark_config = {
        "name": "robovqa_demo",
        "data_dir": "/path/to/robovqa/tfrecords",
        "split": "validation",
        "metrics": ["bleu", "rouge-l"],
        "max_samples": 10  # Small sample for demo
    }
    
    model_config = {
        "name": "mock_demo",
        "response_mode": "echo",
        "generation_kwargs": {
            "temperature": 0.7,
            "max_tokens": 256
        }
    }
    
    # 2. Instantiate benchmark and model
    print("Loading benchmark...")
    benchmark = RoboVQABenchmark(benchmark_config)
    print(f"Loaded {len(benchmark)} samples")
    
    print("\nLoading model...")
    model = MockModel(model_config)
    print(f"Model: {model.name}")
    
    # 3. Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    ground_truths = []
    metadata_list = []
    
    for i, sample in enumerate(benchmark.samples):
        # Preprocess
        sample = benchmark.preprocess(sample)
        
        # Generate prompt
        prompt = benchmark.generate_prompt(sample, model)
        
        # Generate prediction
        prediction = model.generate(
            prompt=prompt,
            images=sample.images,
            videos=sample.videos,
            audios=sample.audios
        )
        
        # Parse output
        prediction = model.parse_output(prediction)
        
        predictions.append(prediction)
        ground_truths.append(sample.answer)
        metadata_list.append(sample.metadata)
        
        print(f"  [{i+1}/{len(benchmark)}] Generated: {prediction[:50]}...")
    
    # 4. Evaluate
    print("\nEvaluating...")
    results = benchmark.evaluate(predictions, ground_truths, metadata_list)
    
    # 5. Display results
    benchmark.print_summary()
    
    # 6. Save results
    output_dir = Path("./demo_results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "results.json"
    benchmark.save_results(results_file)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()

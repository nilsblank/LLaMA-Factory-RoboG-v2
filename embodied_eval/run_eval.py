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

"""Main evaluation runner with Hydra config support."""

import json
from pathlib import Path
from typing import List
import hydra

import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from base import BaseBenchmark, BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# Registry for benchmarks and models
BENCHMARK_REGISTRY = {}
MODEL_REGISTRY = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark class."""
    def decorator(cls):
        BENCHMARK_REGISTRY[name] = cls
        return cls
    return decorator


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# Import and register components
#from . import robovqa_benchmark, models
import robovqa_benchmark
import vstar_benchmark
import robo2vlm_benchmark
import models
# Register benchmarks
register_benchmark("robovqa")(robovqa_benchmark.RoboVQABenchmark)
register_benchmark("vstar")(vstar_benchmark.VStarBenchmark)
register_benchmark("robo2vlm")(robo2vlm_benchmark.Robo2VLMBenchmark)

# Register models
register_model("mock")(models.MockModel)
register_model("llamafactory")(models.LlamaFactoryModel)
register_model("openai")(models.OpenAIModel)
register_model("gemini")(models.GoogleModel)


def instantiate_from_config(cfg: DictConfig):
    """
    Instantiate an object from Hydra config.
    
    Args:
        cfg: Config dict with _target_ key
        
    Returns:
        Instantiated object
    """
    # Check if using _target_ for instantiation
    if "_target_" in cfg:
        #loaded = hydra.utils.instantiate(cfg)  # Ensure Hydra is initialized
        target = cfg._target_
        
        # Import the class
        module_path, class_name = target.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        
        # Remove _target_ from config before passing to __init__
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_dict.pop("_target_", None)
        
        # Instantiate
        return cls(OmegaConf.create(config_dict))
    
    raise ValueError("Config must have _target_ key for instantiation")


def run_single_evaluation(
    benchmark: BaseBenchmark,
    model: BaseModel,
    output_dir: Path,
    save_predictions: bool = True,
    verbose: bool = True,
    batch_size: int = None,
    use_vllm: bool = False,
    vllm_config: dict = None,
    vllm_model_instance: BaseModel = None  # Model instance for vLLM post-processing
) -> dict:
    """
    Run evaluation for a single benchmark-model pair.
    
    Args:
        benchmark: Benchmark instance
        model: Model instance (or None if use_vllm=True)
        output_dir: Directory to save results
        save_predictions: Whether to save predictions to JSONL
        verbose: Whether to print progress
        batch_size: Batch size for batch generation (if model supports it)
        use_vllm: Whether to use vllm_infer for this evaluation
        vllm_config: Config dict for vllm_infer (model_name_or_path, template, etc.)
        vllm_model_instance: Model instance for model-specific processing in vLLM
        
    Returns:
        Evaluation results dict
    """
    if verbose:
        print(f"\n{'='*70}")
        model_name = model.name if model else (vllm_model_instance.name if vllm_model_instance else 'vLLM')
        print(f"Evaluating: {benchmark.name} × {model_name}")
        print(f"{'='*70}")
    
    # Use vLLM inference if requested
    if use_vllm:
        if vllm_config is None:
            raise ValueError("vllm_config required when use_vllm=True")
        
        if verbose:
            print("Using vLLM inference engine")
            if vllm_model_instance:
                print(f"With model-specific processing: {vllm_model_instance.name}")
        
        from .vllm_infer import vllm_infer
        
        # Call vllm_infer with benchmark instance and model instance
        vllm_results = vllm_infer(
            benchmark_instance=benchmark,
            model_instance=vllm_model_instance,  # Pass model for processing
            output_dir=str(output_dir),
            save_predictions=save_predictions,
            save_results=True,
            verbose=verbose,
            batch_size=batch_size or 1024,
            **vllm_config  # model_name_or_path, template, adapter_name_or_path, etc.
        )
        
        return vllm_results["results"]
    
    # Check if model supports batch generation
    has_batch_generate = hasattr(model, 'generate_batch')
    
    if has_batch_generate and batch_size and batch_size > 1:
        # Use efficient batch generation
        if verbose:
            print(f"Using batch generation with batch_size={batch_size}")
        
        # Preprocess all samples
        preprocessed_samples = [benchmark.preprocess(sample) for sample in benchmark.samples]
        
        # Prepare all prompts
        prompts = [benchmark.generate_prompt(sample, model) for sample in preprocessed_samples]
        
        # Prepare multimodal data
        images_list = [sample.images for sample in preprocessed_samples]
        videos_list = [sample.videos for sample in preprocessed_samples]
        audios_list = [sample.audios for sample in preprocessed_samples]
        
        # Batch generate
        if verbose:
            print("Generating predictions in batches...")
        
        predictions = model.generate_batch(
            prompts=prompts,
            images_list=images_list,
            videos_list=videos_list,
            audios_list=audios_list,
            batch_size=batch_size
        )
        
        # Parse outputs
        predictions = [model.parse_output(pred) for pred in predictions]
        
        # Collect ground truths and metadata
        ground_truths = [sample.answer for sample in preprocessed_samples]
        metadata_list = [sample.metadata for sample in preprocessed_samples]
        
    else:
        # Use standard sequential generation
        if verbose:
            print("Using sequential generation")
        
        predictions = []
        ground_truths = []
        metadata_list = []
        
        iterator = tqdm(benchmark.samples, desc="Generating") if verbose else benchmark.samples
        
        for sample in iterator:
            # Preprocess sample
            sample = benchmark.preprocess(sample)
            
            # Generate prompt
            prompt = benchmark.generate_prompt(sample, model)
            
            # Generate prediction
            try:
                prediction = model.generate(
                    prompt=prompt,
                    images=sample.images,
                    videos=sample.videos,
                    audios=sample.audios
                )
            except Exception as e:
                if verbose:
                    print(f"Error generating prediction: {e}")
                prediction = ""
            
            # Parse output if model has custom parser
            prediction = model.parse_output(prediction)
            
            predictions.append(prediction)
            ground_truths.append(sample.answer)
            metadata_list.append(sample.metadata)
    
    # Benchmark-specific post-processing, which can include e.g. denormalization of bboxes
    predictions = [benchmark.postprocess(pred, sample, model) for sample, pred in zip(benchmark.samples, predictions)]

    # Save predictions if requested
    if save_predictions:
        pred_file = output_dir / f"{benchmark.name}_{model.name}_predictions.jsonl"
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pred_file, 'w') as f:
            for pred, gt, meta in zip(predictions, ground_truths, metadata_list):
                f.write(json.dumps({
                    'prediction': pred,
                    'ground_truth': gt,
                    'metadata': meta
                }) + '\n')
        
        if verbose:
            print(f"Saved predictions to: {pred_file}")
    
    # Evaluate
    if verbose:
        print("Computing metrics...")
    
    results = benchmark.evaluate(predictions, ground_truths, metadata_list)
    
    # Save results
    results_file = output_dir / f"{benchmark.name}_{model.name}_results.json"
    benchmark.save_results(results_file)
    
    if verbose:
        print(f"Saved results to: {results_file}")
        benchmark.print_summary()
    
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main evaluation pipeline using Hydra config.
    
    Supports evaluating multiple benchmarks with multiple models.
    All combinations will be evaluated.
    """
    print("="*70)
    print("Embodied Evaluation Framework")
    print("="*70)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup output directory
    output_dir = Path(cfg.get('output_dir', './eval_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate benchmarks
    benchmarks = []
    if 'benchmarks' in cfg:
        if isinstance(cfg.benchmarks, DictConfig):
            #make it a list
            cfg.benchmarks = [cfg.benchmarks]
        for bench_cfg in cfg.benchmarks:
            if bench_cfg is not None:  # Skip None entries
                benchmark = instantiate_from_config(bench_cfg)
                #save the benchmark as jsonl file in sharegpt format

                
                benchmarks.append(benchmark)
                print(f"Loaded benchmark: {benchmark.name} ({len(benchmark)} samples)")
    
    if not benchmarks:
        print("Warning: No benchmarks specified!")
        return
    
    # Instantiate models
    models_list = []
    vllm_models = []  # Track which models should use vLLM
    
    if 'models' in cfg:
        if isinstance(cfg.models, DictConfig):
            cfg.models = [cfg.models]
        for model_cfg in cfg.models:
            if model_cfg is None:  # Skip None entries
                continue
            
            # Check if this is a vLLM model config
            if model_cfg.get('use_vllm', False):
                # Instantiate model class for preprocessing/postprocessing
                # but use vLLM for actual inference
                model_class_target = model_cfg.get('_target_')
                
                if model_class_target:
                    # Instantiate the model (for processing only, not inference)
                    model_instance = instantiate_from_config(model_cfg)
                    print(f"Loaded model class for vLLM processing: {model_instance.name}")
                else:
                    model_instance = None
                    print(f"Registered vLLM model (no model class): {model_cfg.get('name', 'vllm_model')}")
                
                vllm_models.append({
                    'name': model_cfg.get('name', 'vllm_model'),
                    'config': model_cfg,
                    'model_instance': model_instance  # Store for processing
                })
            else:
                # Instantiate normal model
                model = instantiate_from_config(model_cfg)
                models_list.append(model)
                print(f"Loaded model: {model.name}")
    
    if not models_list and not vllm_models:
        print("Warning: No models specified!")
        return
    
    # Run all benchmark-model combinations
    all_results = {}
    
    # Get batch size from config
    batch_size = cfg.get('batch_size', 1)
    
    # ==================== Regular Models ====================
    # Evaluate each benchmark with each regular model
    for benchmark in benchmarks:
        for model in models_list:
            pair_name = f"{benchmark.name}_{model.name}"
            
            try:
                results = run_single_evaluation(
                    benchmark=benchmark,
                    model=model,
                    output_dir=output_dir,
                    save_predictions=cfg.get('save_predictions', True),
                    verbose=cfg.get('verbose', True),
                    batch_size=batch_size,
                    use_vllm=False
                )
                all_results[pair_name] = results
            except Exception as e:
                print(f"Error evaluating {pair_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[pair_name] = {'error': str(e)}
    
    # ==================== vLLM Models ====================
    # KEY OPTIMIZATION: Process all benchmarks with same vLLM model in one go!
    # This avoids expensive vLLM engine reinitialization
    verbose = cfg.get('verbose', True)
    
    for vllm_model in vllm_models:
        try:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Initializing vLLM for model: {vllm_model['name']}")
                print(f"Will process {len(benchmarks)} benchmark(s) with same engine")
                print(f"{'='*70}")
            
            # Extract vLLM config parameters
            vllm_cfg = vllm_model['config']
            vllm_params = {
                'model_name_or_path': vllm_cfg.get('model_name_or_path'),
                'template': vllm_cfg.get('template', 'default'),
                'adapter_name_or_path': vllm_cfg.get('adapter_name_or_path'),
                'temperature': vllm_cfg.get('temperature', 0.95),
                'top_p': vllm_cfg.get('top_p', 0.7),
                'top_k': vllm_cfg.get('top_k', 50),
                'max_new_tokens': vllm_cfg.get('max_new_tokens', 1024),
                'cutoff_len': vllm_cfg.get('cutoff_len', 2048),
                'pipeline_parallel_size': vllm_cfg.get('pipeline_parallel_size', 1),
            }
            
            # Remove None values
            vllm_params = {k: v for k, v in vllm_params.items() if v is not None}
            

            
            # Import and call vllm_infer with ALL benchmarks at once
            from vllm_infer import vllm_infer
            
            bench_results = vllm_infer(
                benchmark_instances=benchmarks,  # Pass all benchmarks!
                model_instance=vllm_model['model_instance'],
                output_dir=str(output_dir),
                save_predictions=cfg.get('save_predictions', True),
                save_results=True,
                verbose=cfg.get('verbose', True),
                batch_size=batch_size or 1024,
                **vllm_params
            )
            
            # Handle both single and multiple benchmark cases
            # When there's 1 benchmark: bench_results is {predictions, results, ...}
            # When there are multiple: bench_results is {benchmark_name: {predictions, results, ...}}
            if len(benchmarks) == 1:
                # Single benchmark case - wrap it in a dict
                bench_name = benchmarks[0].name
                bench_results = {bench_name: bench_results}
            
            # bench_results is now always a dict: {benchmark_name: {predictions, results, ...}}
            for bench_name, bench_result in bench_results.items():
                pair_name = f"{bench_name}_{vllm_model['name']}"
                all_results[pair_name] = bench_result['results']
                
        except FileExistsError as e:
            print(f"Error with vLLM model {vllm_model['name']}: {e}")
            import traceback
            traceback.print_exc()
            # Mark all benchmarks for this model as errored
            for benchmark in benchmarks:
                pair_name = f"{benchmark.name}_{vllm_model['name']}"
                all_results[pair_name] = {'error': str(e)}
    
    # Save combined results
    combined_file = output_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete! Combined results saved to: {combined_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

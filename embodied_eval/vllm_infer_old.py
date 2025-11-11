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

"""
vLLM inference adapted to work with benchmark framework.

This is the STANDARD way to run inference - uses efficient vLLM batch processing
while working seamlessly with the benchmark classes.
"""

import gc
import json
from pathlib import Path
from typing import Optional

import fire
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

import sys


from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import ModelArguments, DataArguments
from llamafactory.model import load_tokenizer

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    # Model arguments (required)
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    template: str = "default",
    
    # Benchmark arguments - can be single or list
    benchmark_instances: Optional[list] = None,  # List of benchmark instances
    benchmark_configs: Optional[list] = None,  # List of paths to benchmark YAML configs
    benchmark_names: Optional[list] = None,  # List of benchmark names
    
    # For backward compatibility - single benchmark
    benchmark_instance: Optional[object] = None,
    benchmark_config: Optional[str] = None,
    benchmark_name: str = "robovqa",
    benchmark_data_dir: Optional[str] = None,
    benchmark_split: str = "validation",
    
    # Model instance for post-processing (optional)
    model_instance: Optional[object] = None,  # BaseModel instance for parse_output, process_bbox, etc.
    
    # vLLM engine arguments
    cutoff_len: int = 2048,
    max_new_tokens: int = 1024,
    pipeline_parallel_size: int = 1,
    vllm_config: str = "{}",
    
    # Generation arguments
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
    
    # Multimodal arguments
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    
    # Processing arguments
    batch_size: int = 1024,
    max_samples: Optional[int] = None,
    skip_special_tokens: bool = True,
    
    # Output arguments
    output_dir: str = "./eval_results",
    save_name: Optional[str] = None,  # Auto-generated if None
    save_predictions: bool = True,
    save_results: bool = True,
    verbose: bool = True,
    force_eager = True
):
    """
    Perform batch generation using vLLM engine with benchmark framework.
    
    This is the STANDARD inference method - combines efficient vLLM batch processing
    with the flexible benchmark framework.
    
    Usage:
        # Using benchmark config file
        python -m embodied_eval.vllm_infer \\
            --model_name_or_path /path/to/model \\
            --benchmark_config embodied_eval/configs/benchmarks/robovqa.yaml \\
            --batch_size 1024
        
        # Using benchmark name and overrides    
        python -m embodied_eval.vllm_infer \\
            --model_name_or_path /path/to/model \\
            --benchmark_name robovqa \\
            --benchmark_data_dir /path/to/data \\
            --benchmark_split validation \\
            --batch_size 1024
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of GPUs.")
    
    if verbose:
        print("="*70)
        print("vLLM Inference with Benchmark Framework")
        print("="*70)
    
    # ==================== Load Benchmarks ====================
    if verbose:
        print("\n[1/5] Loading benchmarks...")
    
    # Collect all benchmarks to process
    benchmarks = []
    
    # Handle multiple benchmarks (new way - efficient!)
    if benchmark_instances is not None:
        benchmarks = benchmark_instances if isinstance(benchmark_instances, list) else [benchmark_instances]
        if verbose:
            for bench in benchmarks:
                print(f"   Using provided benchmark: {bench.name} ({len(bench)} samples)")
    
    # Single benchmark for backward compatibility
    elif benchmark_instance is not None:
        benchmarks = [benchmark_instance]
        if verbose:
            print(f"   Using provided benchmark: {benchmark_instance.name} ({len(benchmark_instance)} samples)")
    
    # Load from configs/names
    else:
        from embodied_eval.run_eval import BENCHMARK_REGISTRY
        
        # Determine which benchmarks to load
        bench_specs = []
        if benchmark_configs:
            bench_specs = [{'type': 'config', 'value': cfg} for cfg in (benchmark_configs if isinstance(benchmark_configs, list) else [benchmark_configs])]
        elif benchmark_names:
            bench_specs = [{'type': 'name', 'value': name} for name in (benchmark_names if isinstance(benchmark_names, list) else [benchmark_names])]
        else:
            # Default to single benchmark
            bench_specs = [{'type': 'name', 'value': benchmark_name}]
        
        # Load each benchmark
        for spec in bench_specs:
            if spec['type'] == 'config':
                bench_cfg = OmegaConf.load(spec['value'])
            else:
                bench_cfg = OmegaConf.create({
                    "name": spec['value'],
                    "data_dir": benchmark_data_dir or f"/path/to/{spec['value']}/data",
                    "split": benchmark_split,
                    "metrics": ["bleu", "rouge-l"],
                })
            
            # Override with CLI arguments
            if benchmark_data_dir:
                bench_cfg.data_dir = benchmark_data_dir
            if max_samples:
                bench_cfg.max_samples = max_samples
            
            # Get benchmark class
            bench_name = bench_cfg.get('name', spec['value'])
            if bench_name not in BENCHMARK_REGISTRY:
                raise ValueError(f"Benchmark '{bench_name}' not found. Available: {list(BENCHMARK_REGISTRY.keys())}")
            
            benchmark_cls = BENCHMARK_REGISTRY[bench_name]
            benchmark = benchmark_cls(bench_cfg)
            benchmarks.append(benchmark)
            
            if verbose:
                print(f"   Loaded benchmark: {benchmark.name} ({len(benchmark)} samples)")
    
    if not benchmarks:
        raise ValueError("No benchmarks provided!")
    
    if verbose:
        total_samples = sum(len(b) for b in benchmarks)
        print(f"   Total: {len(benchmarks)} benchmark(s), {total_samples} samples")
    
    # ==================== Initialize vLLM ====================
    if verbose:
        print("\n[2/5] Initializing vLLM engine...")
    
    # Create model args
    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=[adapter_name_or_path] if adapter_name_or_path else None,
        vllm_config=eval(vllm_config) if isinstance(vllm_config, str) else vllm_config,
    )
    
    # Create data args for tokenizer
    data_args = DataArguments(
        template=template,
        cutoff_len=cutoff_len,
    )
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # For vLLM generate
    
    # Build vLLM engine args
    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    
    is_debugging = False
    if force_eager:
        is_debugging = True
    else:
        # Check for VS Code debugger (debugpy)
        if "debugpy" in sys.modules:
            try:
                import debugpy
                if debugpy.is_client_connected():
                    is_debugging = True
            except ImportError:
                pass  # debugpy not installed
    
    # Fallback to standard trace check
    if not is_debugging:
        is_debugging = sys.gettrace() is not None
    if force_eager or is_debugging:
        if verbose:
            if force_eager:
                print("   [!] Eager mode forced via --force_eager flag (enforce_eager=True).")
            else:
                print("   [!] Debugger detected. Forcing eager mode (enforce_eager=True) to prevent torch.compile errors.")
        engine_args["enforce_eager"] = True
    # Add multimodal limits
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 10, "video": 2, "audio": 2}
    
    # Update with custom vllm config
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
    
    # Initialize LLM
    llm = LLM(**engine_args)
    
    if verbose:
        print(f"   Model: {model_name_or_path}")
        print(f"   Tensor parallel size: {engine_args['tensor_parallel_size']}")
        print(f"   Pipeline parallel size: {engine_args['pipeline_parallel_size']}")
    
    # ==================== Prepare Sampling Params ====================
    sampling_params = SamplingParams(
        repetition_penalty=repetition_penalty or 1.0,
        temperature=temperature,
        top_p=top_p or 1.0,
        top_k=top_k or -1,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    
    # LoRA request
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None
    
    # ==================== Process All Benchmarks ====================
    # Key optimization: Reuse vLLM engine for all benchmarks!
    
    all_benchmark_results = {}
    
    for benchmark in benchmarks:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing benchmark: {benchmark.name}")
            print(f"{'='*70}")
        
        # ==================== Prepare Data ====================
        if verbose:
            print("\n[3/5] Preparing data for batch inference...")
        
        # Preprocess all samples and prepare prompts
        all_samples = []
        all_prompts = []
        
        if verbose and model_instance is not None:
            print(f"   Using model-specific preprocessing: {model_instance.name}")
        
        for sample in benchmark.samples:
            # Preprocess sample (benchmark preprocessing first)
            sample = benchmark.preprocess(sample)
            
            # Apply model-specific preprocessing if model instance provided
            if model_instance is not None:
                # Process images if present
                if sample.images is not None:
                    sample.images = [model_instance.process_image(img) for img in sample.images]
                
                # Process bboxes in metadata if present
                if sample.metadata and 'bbox' in sample.metadata:
                    bbox = sample.metadata['bbox']
                    original_size = sample.metadata.get('original_size', (1000, 1000))
                    target_size = sample.metadata.get('target_size', original_size)
                    bbox_format = sample.metadata.get('bbox_format', 'xyxy')
                    
                    processed_bbox = model_instance.process_bbox(
                        bbox, original_size, target_size, bbox_format
                    )
                    sample.metadata['bbox_processed'] = processed_bbox
            
            all_samples.append(sample)
            
            # Generate prompt (benchmark knows how to format it)
            # Pass model instance so benchmark can use model-specific formatting
            prompt = benchmark.generate_prompt(sample, model_instance)
            if isinstance(prompt, dict):
                prompt_text = prompt.get('content', str(prompt))
            else:
                prompt_text = prompt
            all_prompts.append(prompt_text)
        
        if verbose:
            print(f"   Prepared {len(all_prompts)} prompts")
        
        # ==================== Batch Inference ====================
        if verbose:
            print("\n[4/5] Running vLLM batch inference...")
        
        all_predictions = []
        all_ground_truths = []
        all_metadata = []
        
        # Flag to print model processing message once
        printed_model_processing = False
        
        # Process in batches
        for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches", disable=not verbose):
            vllm_inputs = []
            batch_samples = all_samples[i : min(i + batch_size, len(all_samples))]
            batch_prompts = all_prompts[i : min(i + batch_size, len(all_prompts))]
            
            for j, (sample, prompt_text) in enumerate(zip(batch_samples, batch_prompts)):
                # Tokenize prompt
                prompt_ids = tokenizer.encode(prompt_text)
                
                # Prepare multimodal data
                multi_modal_data = None
                if sample.images is not None and len(sample.images) > 0:
                    multi_modal_data = {
                        "image": template_obj.mm_plugin._regularize_images(
                            sample.images,
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels
                        )["images"]
                    }
                elif sample.videos is not None and len(sample.videos) > 0:
                    multi_modal_data = {
                        "video": template_obj.mm_plugin._regularize_videos(
                            sample.videos,
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels,
                            video_fps=video_fps,
                            video_maxlen=video_maxlen,
                        )["videos"]
                    }
                elif sample.audios is not None and len(sample.audios) > 0:
                    audio_data = template_obj.mm_plugin._regularize_audios(
                        sample.audios,
                        sampling_rate=16000,
                    )
                    multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
                
                vllm_inputs.append({
                    "prompt_token_ids": prompt_ids,
                    "multi_modal_data": multi_modal_data
                })
            
            # Generate batch
            results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
            preds = [result.outputs[0].text for result in results]
            
            # Apply model-specific post-processing if model instance provided
            if model_instance is not None:
                if verbose and not printed_model_processing:
                    print(f"   Applying model-specific post-processing ({model_instance.name})")
                    printed_model_processing = True
                preds = [model_instance.parse_output(pred) for pred in preds]
            
            # Collect results
            all_predictions.extend(preds)
            all_ground_truths.extend([s.answer for s in batch_samples])
            all_metadata.extend([s.metadata for s in batch_samples])
            
            gc.collect()
        
        if verbose:
            print(f"   Generated {len(all_predictions)} predictions")
        
        # ==================== Evaluate & Save ====================
        if verbose:
            print("\n[5/5] Evaluating and saving results...")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-generate save name if not provided
        model_name = Path(model_name_or_path).name
        bench_save_name = f"{benchmark.name}_{model_name}_predictions.jsonl"
        
        # Save predictions
        if save_predictions:
            pred_file = output_path / bench_save_name
            with open(pred_file, "w", encoding="utf-8") as f:
                for pred, gt, meta in zip(all_predictions, all_ground_truths, all_metadata):
                    f.write(json.dumps({
                        "prediction": pred,
                        "ground_truth": gt,
                        "metadata": meta
                    }, ensure_ascii=False) + "\n")
            
            if verbose:
                print(f"   Saved predictions to: {pred_file}")
        
        # Evaluate
        results = benchmark.evaluate(all_predictions, all_ground_truths, all_metadata)
        
        # Save results
        if save_results:
            results_file = output_path / bench_save_name.replace("_predictions.jsonl", "_results.json")
            benchmark.save_results(results_file)
            if verbose:
                print(f"   Saved results to: {results_file}")
        
        # Print summary
        if verbose:
            benchmark.print_summary()
        
        # Store results for this benchmark
        all_benchmark_results[benchmark.name] = {
            "predictions": all_predictions,
            "ground_truths": all_ground_truths,
            "metadata": all_metadata,
            "results": results
        }
    
    # ==================== Final Summary ====================
    if verbose:
        print("\n" + "="*70)
        print(f"Inference complete!")
        print(f"Processed {len(benchmarks)} benchmark(s) with same vLLM engine")
        for bench_name, bench_res in all_benchmark_results.items():
            print(f"  - {bench_name}: {len(bench_res['predictions'])} predictions")
        print("="*70)
    
    # Return results for programmatic use
    # If single benchmark, return its results for backward compatibility
    if len(benchmarks) == 1:
        return all_benchmark_results[benchmarks[0].name]
    else:
        # Multiple benchmarks - return all results
        return all_benchmark_results


if __name__ == "__main__":
    fire.Fire(vllm_infer)

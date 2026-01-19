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

This version uses Hugging Face's `apply_chat_template` for preprocessing
and passes raw multimodal data to vLLM, removing LlamaFactory's
custom data loaders. It also supports per-request processor arguments
(like `fps`) and global processor arguments (like `max_pixels`) via vllm_config.
"""

import gc
import json
import sys
from pathlib import Path
from typing import Optional

import fire
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

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
    force_eager: bool = False
):
    """
    Perform batch generation using vLLM engine with benchmark framework.
    
    This version uses Hugging Face's `apply_chat_template` for preprocessing.
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of GPUs.")
    
    if verbose:
        print("="*70)
        print("vLLM Inference with Benchmark Framework (HF Chat Template)")
        print("="*70)
    
    # ==================== Load Benchmarks ====================
    if verbose:
        print("\n[1/5] Loading benchmarks...")
    
    # Collect all benchmarks to process
    benchmarks = []
    
    # Handle multiple benchmarks (new way - efficient!)
    if benchmark_instances is not None:
        benchmarks = benchmark_instances if isinstance(benchmark_instances, list) else [benchmark_instances]
    else:
        raise ValueError("Please provide benchmark_instances as a list of benchmark instances.")


    
    if not benchmarks:
        raise ValueError("No valid benchmarks were loaded!")
    
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
    # We still get template_obj for stop tokens, but won't use it for preprocessing
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Build vLLM engine args
    #infer with bf16
    infer_dtype = "bfloat16"
    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 4096,
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

    # Update with custom vllm config
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
    
    # Initialize LLM

    
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
    
    llm = None
    
    # LoRA request
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None
    
    # ==================== Process All Benchmarks ====================
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
        all_chat_messages = []  # Store chat message lists
        
        if verbose and model_instance is not None:
            print(f"   Using model-specific preprocessing: {model_instance.name}")
        
        for sample in benchmark.samples:
            # Preprocess sample (benchmark preprocessing first)

            sample = benchmark.preprocess(sample)
            
            # Apply model-specific preprocessing if model instance provided
            if model_instance is not None:
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
            # **ASSUMPTION**: This returns a chat list, e.g.:
            # [{"role": "user", "content": "What is in this image? <image>"}]
            chat = benchmark.generate_prompt(sample, model_instance)
            all_chat_messages.append(chat)
        
        

        if verbose:
            print(f"   Prepared {len(all_chat_messages)} chat prompts")
        
        # ==================== Batch Inference ====================
        if verbose:
            print("\n[4/5] Running vLLM batch inference...")
        
        all_predictions = []
        all_ground_truths = []
        all_metadata = []
        
        # Flag to print model processing message once
        printed_model_processing = False
        
        model_name = Path(model_name_or_path).name

        model_name = model_instance.config.name if model_instance is not None else model_name

        bench_save_name = f"{benchmark.name}_{model_name}_predictions.jsonl"
        
        pred_file = Path(output_dir) / bench_save_name
        
        
        
        #check if exists, then load from file
        if pred_file.exists():
            if verbose:
                print(f"   Predictions file already exists: {pred_file}")
                print(f"   Loading existing predictions...")
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    all_predictions.append(item["prediction"])
                    all_ground_truths.append(item["ground_truth"])
                    all_metadata.append(item.get("metadata", {}))
            
            for m_idx, meta in enumerate(all_metadata):
                #check if image_paths in meta, if not overwrite meta from benchmark samples
                if "image_paths" not in meta or not meta["image_paths"]:
                    sample_meta = benchmark.samples[m_idx].metadata
                    if "image_paths" in sample_meta:
                        all_metadata[m_idx]["image_paths"] = sample_meta["image_paths"]
            
            if verbose:
                print(f"   Loaded {len(all_predictions)} existing predictions.")
            # Skip to evaluation
            results = benchmark.evaluate(all_predictions, all_ground_truths, all_metadata)
            if save_results:
                results_file = Path(output_dir) / bench_save_name.replace("_predictions.jsonl", "_results.json")
                benchmark.save_results(results_file)
                if verbose:
                    print(f"   Saved results to: {results_file}")
            benchmark.print_summary()
            all_benchmark_results[benchmark.name] = {
                "predictions": all_predictions,
                "ground_truths": all_ground_truths,
                "metadata": all_metadata,
                "results": results
            }
            continue
        
        if llm is None:
            llm = LLM(**engine_args)
    
        if verbose:
            print(f"   Model: {model_name_or_path}")
            print(f"   Tensor parallel size: {engine_args['tensor_parallel_size']}")
            print(f"   Pipeline parallel size: {engine_args['pipeline_parallel_size']}")
            if 'mm_processor_kwargs' in engine_args:
                print(f"   Global MM Processor Kwargs: {engine_args['mm_processor_kwargs']}")
            
        # Process in batches
        for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches", disable=not verbose):
            vllm_inputs = []
            batch_samples = all_samples[i : min(i + batch_size, len(all_samples))]
            batch_chats = all_chat_messages[i : min(i + batch_size, len(all_samples))]
            
            for j, (sample, chat) in enumerate(zip(batch_samples, batch_chats)):
                
                # Use model's prepare_vllm_inputs_from_chat method if available
                if model_instance is not None and hasattr(model_instance, 'prepare_vllm_inputs_from_chat'):
                    vllm_input = model_instance.prepare_vllm_inputs_from_chat(chat, sample)
                    vllm_inputs.append(vllm_input)
                else:
                    raise ValueError("Model instance with method 'prepare_vllm_inputs_from_chat' must be provided.")

                        
            
            # Generate batch
            if not vllm_inputs:
                continue # Skip if batch is empty due to errors
                
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

        model_name = model_instance.config.name if model_instance is not None else model_name

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
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

"""MotionBench benchmark implementation - Foundation Motion Understanding Benchmark."""

import json
import os
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from base import BaseBenchmark, Sample


class MotionBenchBenchmark(BaseBenchmark):
    """
    MotionBench: Foundation Motion Understanding Benchmark.
    
    Evaluates vision-language models on motion understanding tasks using video QA.
    Supports concurrent processing with progress tracking.
    
    Features:
    - Loads video and QA pairs from JSONL metadata
    - Parallel video processing with configurable workers
    - Accuracy metrics computation
    - Result caching and resumption
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """
        Initialize MotionBench benchmark.
        
        Args:
            cfg: Configuration containing:
                - name: Benchmark name (default: "motionbench")
                - data_dir: Path to MotionBench dataset root
                - video_base_path: Path to video directory (default: data_dir/public-dataset)
                - metadata_path: Path to metadata JSONL file
                  (default: data_dir/MotionBench/video_info.meta.jsonl)
                - max_workers: Number of parallel workers (default: 8)
                - cache_dir: Directory for caching results (default: ./motionbench_cache)
                - resume: Whether to resume from cached results (default: true)
        """
        super().__init__(cfg)
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_keys = ["data_dir"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Config missing required key: {key}")
        
        # Set defaults for optional parameters
        if "video_base_path" not in self.config:
            self.config.video_base_path = os.path.join(
                self.config.data_dir, "public-dataset"
            )
        
        if "metadata_path" not in self.config:
            self.config.metadata_path = os.path.join(
                self.config.data_dir, "MotionBench", "video_info.meta.jsonl"
            )
        
        if "max_workers" not in self.config:
            self.config.max_workers = 8
        
        if "cache_dir" not in self.config:
            self.config.cache_dir = "./motionbench_cache"
        
        if "resume" not in self.config:
            self.config.resume = True
    
    def _load_data(self) -> None:
        """Load and parse the MotionBench dataset from JSONL metadata file."""
        metadata_path = self.config.metadata_path
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"MotionBench metadata file not found at: {metadata_path}\n"
                f"Please download the dataset or provide correct path."
            )
        
        self.samples = []
        self._raw_video_info = []
        
        # Load JSONL metadata
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    video_info = json.loads(line.strip())
                    self._raw_video_info.append(video_info)
                    
                    # Create samples for each QA pair
                    video_path = os.path.join(
                        self.config.video_base_path,
                        video_info.get("video_path", "")
                    )
                    
                    if not os.path.exists(video_path):
                        continue
                    
                    questions = video_info.get("qa", [])
                    for question in questions:
                        sample = Sample(
                            question=question.get("question", ""),
                            answer=question.get("answer", ""),
                            metadata={
                                "video_path": video_path,
                                "original_video_path": video_info.get("video_path", ""),
                                "question_id": question.get("id", ""),
                            }
                        )
                        self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} QA samples from {len(self._raw_video_info)} videos")
    
    def preprocess(self, sample: Sample) -> Sample:
        """
        Preprocess a single sample.
        
        For MotionBench, this ensures video path is valid and question/answer
        are properly formatted.
        
        Args:
            sample: Input sample
            
        Returns:
            Preprocessed sample
        """
        # Video path should already be in metadata
        video_path = sample.metadata.get("video_path", "")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Ensure question and answer are strings
        sample.question = str(sample.question).strip()
        sample.answer = str(sample.answer).strip()
        
        return sample
    
    def evaluate(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truths using accuracy.
        
        Uses a simple matching strategy:
        - Extract first choice letter (A, B, C, D) from prediction
        - Compare with ground truth answer
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers
            metadata: Optional list of metadata dicts
            
        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy (0-1)
                - correct: Number of correct predictions
                - total: Total number of predictions
                - per_video: Per-video accuracy breakdown
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground_truths ({len(ground_truths)}) "
                "must have the same length"
            )
        
        correct = 0
        total = len(predictions)
        per_video = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred, truth, meta in zip(predictions, ground_truths, metadata or [None] * len(predictions)):
            score = self._check_answer(pred, truth)
            correct += score
            
            # Track per-video accuracy if metadata available
            if meta and "video_path" in meta:
                video_path = meta["video_path"]
                per_video[video_path]["correct"] += score
                per_video[video_path]["total"] += 1
        
        # Compute per-video accuracies
        per_video_accuracy = {}
        for video_path, counts in per_video.items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            per_video_accuracy[video_path] = {
                "accuracy": round(acc, 4),
                "correct": counts["correct"],
                "total": counts["total"]
            }
        
        self.results = {
            "accuracy": round(correct / total, 4) if total > 0 else 0,
            "correct": correct,
            "total": total,
            "per_video": per_video_accuracy
        }
        
        return self.results
    
    def _check_answer(self, prediction: str, answer: str) -> int:
        """
        Check if prediction matches the answer.
        
        Simple matching logic:
        - Convert both to lowercase
        - Extract first sentence of prediction (before punctuation)
        - Check if answer appears in prediction
        - Special case: answer "na" always matches
        
        Args:
            prediction: Model prediction
            answer: Ground truth answer
            
        Returns:
            1 if match, 0 otherwise
        """
        answer_lower = answer.lower().strip()
        
        # Special case: "na" (not applicable) always matches
        if answer_lower == "na":
            return 1
        
        # Extract first sentence from prediction
        pred_lower = prediction.lower()
        # Take text before first punctuation or comma
        pred_first = pred_lower.split(":")[0].split(",")[0].split(".")[0].strip()
        
        # Check if answer is in first part of prediction
        if answer_lower in pred_first:
            return 1
        
        return 0
    
    def process_in_parallel(
        self,
        video_processor_fn,
        max_workers: Optional[int] = None,
        resume: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all videos in parallel using provided function.
        
        Args:
            video_processor_fn: Function that takes (sample, metadata) and returns result
                               Should return dict or None
            max_workers: Number of parallel workers (default: config.max_workers)
            resume: Whether to resume from cache (default: True)
            
        Returns:
            Dictionary mapping video paths to list of processed results
        """
        if max_workers is None:
            max_workers = self.config.get("max_workers", 8)
        
        # Setup cache directory
        cache_dir = Path(self.config.get("cache_dir", "./motionbench_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cached results if resuming
        cache_file = cache_dir / f"{self.name}_results.json"
        results = {}
        processed_count = 0
        
        if resume and cache_file.exists():
            with open(cache_file, 'r') as f:
                results = json.load(f)
            processed_count = sum(len(v) for v in results.values())
            print(f"Resuming from cache: {processed_count} results already processed")
        
        # Prepare tasks for unprocessed samples
        tasks_to_process = []
        for sample in self.samples:
            video_path = sample.metadata.get("video_path", "")
            if video_path not in results:
                results[video_path] = []
            
            # Check if this specific question is already processed
            question_id = sample.metadata.get("question_id", "")
            if not any(r.get("question_id") == question_id for r in results[video_path]):
                tasks_to_process.append((sample, sample.metadata))
        
        if not tasks_to_process:
            print("All samples already processed. Using cached results.")
            return results
        
        print(f"Processing {len(tasks_to_process)} samples with {max_workers} workers")
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(video_processor_fn, sample, meta): (sample, meta)
                for sample, meta in tasks_to_process
            }
            
            with tqdm(total=len(tasks_to_process), desc="Processing videos") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    sample, meta = futures[future]
                    video_path = meta.get("video_path", "")
                    
                    try:
                        result = future.result()
                        if result is not None:
                            result["question_id"] = meta.get("question_id", "")
                            if video_path not in results:
                                results[video_path] = []
                            results[video_path].append(result)
                    except Exception as e:
                        print(f"Error processing {video_path}: {str(e)}")
                    
                    pbar.update(1)
                    
                    # Save intermediate results
                    if (processed_count + len(tasks_to_process) - sum(1 for v in futures.values())) % 10 == 0:
                        self._save_cache(results, cache_file)
        
        # Save final results
        self._save_cache(results, cache_file)
        
        return results
    
    def _save_cache(self, results: Dict, cache_file: Path) -> None:
        """
        Save results to cache file atomically.
        
        Args:
            results: Results dictionary
            cache_file: Path to cache file
        """
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix('.json.temp')
        
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        temp_file.replace(cache_file)
    
    def get_evaluation_prompts(self) -> List[str]:
        """
        Get standardized evaluation prompts for MotionBench.
        
        Returns:
            List of prompt templates
        """
        return [
            "{question}\nPlease directly output the choice (A, B, C, D). No other text.",
            "{question}\nAnswer: ",
            "Question: {question}\nChoose from A, B, C, D: ",
        ]
    
    def print_summary(self) -> None:
        """Print a summary of evaluation results with per-video breakdown."""
        if not self.results:
            print("No results to display.")
            return
        
        print(f"\n{'='*80}")
        print(f"MotionBench Evaluation Results")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {self.results['accuracy']:.4f}")
        print(f"Correct: {self.results['correct']} / {self.results['total']}")
        
        if self.results.get("per_video"):
            print(f"\n{'Video Path':<50} {'Accuracy':<12} {'Correct':<10}")
            print("-" * 72)
            for video_path, metrics in self.results["per_video"].items():
                video_name = Path(video_path).name
                print(
                    f"{video_name:<50} "
                    f"{metrics['accuracy']:<12.4f} "
                    f"{metrics['correct']}/{metrics['total']:<8}"
                )
        
        print(f"{'='*80}\n")

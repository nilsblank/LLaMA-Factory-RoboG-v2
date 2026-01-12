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

"""FoundationMotion benchmark implementation for hand gesture and motion understanding."""

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


class FoundationMotionBenchmark(BaseBenchmark):
    """
    FoundationMotion: Hand Gesture and Motion Understanding Benchmark.
    
    Evaluates vision-language models on hand gesture and motion understanding tasks.
    Supports concurrent video processing with configurable workers.
    
    Dataset structure:
    - videos/: Contains .mp4 video files organized by task
    - qa_shuffled/: Contains .json QA files with corresponding video names
    
    QA file format:
    ```json
    {
      "human": {
        "questions": [
          {
            "question": "What is the man doing with his hands?",
            "A": "Using one hand to gesture.",
            "B": "Both hands are not visible...",
            "C": "Holding an object in his right hand.",
            "D": "Holding an object in his left hand.",
            "correct_answer": "B"
          }
        ]
      }
    }
    ```
    
    Or with GPT-generated QA:
    ```json
    {
      "gpt4o_mini_res": {
        "questions": [...]
      }
    }
    ```
    
    Features:
    - Automatic video enumeration from directory
    - Matched QA pair loading
    - Parallel video processing with progress tracking
    - Accuracy metrics computation
    - Result caching and resumption
    - Atomic file writes for safety
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """
        Initialize FoundationMotion benchmark.
        
        Args:
            cfg: Configuration containing:
                - name: Benchmark name (default: "foundation_motion")
                - data_dir: Path to benchmark dataset root
                - task: Task name (used to construct paths) (default: "av_hands_eval")
                - video_base_path: Path to videos directory
                  (default: data_dir/{task}/videos)
                - qa_base_path: Path to QA directory
                  (default: data_dir/{task}/qa_shuffled)
                - max_workers: Number of parallel workers (default: 8)
                - cache_dir: Directory for caching results (default: ./foundation_motion_cache)
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
        if "task" not in self.config:
            self.config.task = "av_hands_eval"
        
        if "video_base_path" not in self.config:
            self.config.video_base_path = os.path.join(
                self.config.data_dir, self.config.task, "videos"
            )
        
        if "qa_base_path" not in self.config:
            self.config.qa_base_path = os.path.join(
                self.config.data_dir, self.config.task, "qa_shuffled"
            )
        
        if "max_workers" not in self.config:
            self.config.max_workers = 8
        
        if "cache_dir" not in self.config:
            self.config.cache_dir = "./foundation_motion_cache"
        
        if "resume" not in self.config:
            self.config.resume = True
    
    def _load_data(self) -> None:
        """Load and parse the FoundationMotion dataset."""
        video_base_path = self.config.video_base_path
        qa_base_path = self.config.qa_base_path
        
        if not os.path.exists(video_base_path):
            raise FileNotFoundError(
                f"Video directory not found at: {video_base_path}"
            )
        
        if not os.path.exists(qa_base_path):
            raise FileNotFoundError(
                f"QA directory not found at: {qa_base_path}"
            )
        
        self.samples = []
        self._video_qa_pairs = {}
        
        # Enumerate all .mp4 files
        mp4_files = self._enumerate_mp4_files(video_base_path)
        print(f"Found {len(mp4_files)} video files")
        
        # Load QA pairs for each video
        for video_path in mp4_files:
            # Construct corresponding QA file path
            relative_path = os.path.relpath(video_path, video_base_path)
            qa_path = os.path.join(qa_base_path, relative_path.replace(".mp4", ".json"))
            
            if not os.path.exists(qa_path):
                print(f"Warning: QA file not found for {video_path}")
                continue
            
            try:
                with open(qa_path, 'r') as f:
                    qa_data = json.load(f)
                
                # Extract questions - try both "human" and "gpt4o_mini_res" keys
                questions = None
                if "human" in qa_data:
                    questions = qa_data["human"].get("questions", [])
                elif "gpt4o_mini_res" in qa_data:
                    questions = qa_data["gpt4o_mini_res"].get("questions", [])
                else:
                    # Try to find any key with "questions"
                    for key in qa_data.keys():
                        if isinstance(qa_data[key], dict) and "questions" in qa_data[key]:
                            questions = qa_data[key]["questions"]
                            break
                
                if not questions:
                    print(f"Warning: No questions found in {qa_path}")
                    continue
                
                self._video_qa_pairs[video_path] = questions
                
                # Create samples for each QA pair
                for q_idx, question in enumerate(questions):
                    # Extract answer - try both "correct_answer" and "answer" keys
                    answer = question.get("correct_answer") or question.get("answer", "")
                    
                    # Build multiple choice string from A, B, C, D
                    choices_text = "\n".join([
                        f"{k}: {question.get(k, '')}"
                        for k in ["A", "B", "C", "D"]
                        if k in question
                    ])
                    
                    sample = Sample(
                        question=question.get("question", ""),
                        answer=answer,
                        metadata={
                            "video_path": video_path,
                            "qa_path": qa_path,
                            "relative_path": relative_path,
                            "question_index": q_idx,
                            "choices": choices_text,
                            "full_question_data": question
                        }
                    )
                    self.samples.append(sample)
            
            except json.JSONDecodeError as e:
                print(f"Error parsing QA file {qa_path}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} QA samples from {len(self._video_qa_pairs)} videos")
    
    def _enumerate_mp4_files(self, directory: str) -> List[str]:
        """
        Enumerate all .mp4 files in directory recursively.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of paths to .mp4 files
        """
        mp4_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))
        
        return sorted(mp4_files)
    
    def preprocess(self, sample: Sample) -> Sample:
        """
        Preprocess a single sample.
        
        For FoundationMotion, this ensures video path is valid and question/answer
        are properly formatted.
        
        Args:
            sample: Input sample
            
        Returns:
            Preprocessed sample
        """
        # Validate video path
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
        
        Uses answer matching strategy:
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
        
        Matching logic:
        - Convert both to lowercase
        - Extract first sentence of prediction (before punctuation/comma)
        - Check if answer appears in first part
        
        Args:
            prediction: Model prediction
            answer: Ground truth answer
            
        Returns:
            1 if match, 0 otherwise
        """
        answer_lower = answer.lower().strip()
        
        # Special case: "na", "n/a" (not applicable) always matches
        if answer_lower in ("na", "n/a", "n.a", "not applicable"):
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
                               Should return dict with at least:
                               - output: Model prediction
                               - score: 0 or 1
            max_workers: Number of parallel workers (default: config.max_workers)
            resume: Whether to resume from cache (default: True)
            
        Returns:
            Dictionary mapping video paths to list of processed results
        """
        if max_workers is None:
            max_workers = self.config.get("max_workers", 8)
        
        # Setup cache directory
        cache_dir = Path(self.config.get("cache_dir", "./foundation_motion_cache"))
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
            q_idx = sample.metadata.get("question_index", "")
            if not any(r.get("question_index") == q_idx for r in results[video_path]):
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
                            result["question_index"] = meta.get("question_index", "")
                            if video_path not in results:
                                results[video_path] = []
                            results[video_path].append(result)
                    except Exception as e:
                        print(f"Error processing {video_path}: {str(e)}")
                    
                    pbar.update(1)
                    
                    # Save intermediate results
                    if len(tasks_to_process) > 0:
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
        Get standardized evaluation prompts for FoundationMotion.
        
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
        print(f"FoundationMotion Evaluation Results")
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
